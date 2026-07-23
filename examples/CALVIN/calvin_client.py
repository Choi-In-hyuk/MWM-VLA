"""CALVIN evaluation client. Wraps our WebSocket policy server into the
CalvinBaseModel interface expected by calvin_agent.evaluation.evaluate_policy.

Env-provided observation dict:
  obs["rgb_obs"]["rgb_static"]   : (200,200,3) uint8
  obs["rgb_obs"]["rgb_gripper"]  : (84,84,3)   uint8
  obs["robot_obs"]               : (15,)        float

The wrapper:
  - resizes both views to 256x256
  - packs into the same request format our server_policy expects (LIBERO client parity)
  - maintains a per-subtask action-chunk buffer so the server is only queried
    every H (=7) steps; between queries we play back the remaining actions.
"""
from __future__ import annotations
from collections import deque
from typing import Optional
import numpy as np
import cv2 as cv

from calvin_agent.models.calvin_base_model import CalvinBaseModel
from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy


CALVIN_IMG_SIZE = 256   # our model expects 256x256 per view (LIBERO parity)
CALVIN_ACTION_HORIZON = 7


def _to_rgb256(img: np.ndarray) -> np.ndarray:
    """Resize (H,W,3) uint8 image to 256x256."""
    if img.shape[:2] != (CALVIN_IMG_SIZE, CALVIN_IMG_SIZE):
        img = cv.resize(img, (CALVIN_IMG_SIZE, CALVIN_IMG_SIZE), interpolation=cv.INTER_AREA)
    return img.astype(np.uint8)


def _robot_obs_15d_to_8d(robot_obs_15d: np.ndarray) -> np.ndarray:
    """CALVIN robot_obs [EE_pos(3), EE_ori(3), gripper_width(1), joints(7), gripper_action(1)]
       -> our 8D state [EE_pos(3), EE_ori(3), gripper_width(1), gripper_action(1)]."""
    r = np.asarray(robot_obs_15d, dtype=np.float32)
    return np.concatenate([r[0:6], r[6:7], r[14:15]]).astype(np.float32)


class OurCalvinModel(CalvinBaseModel):
    """CalvinBaseModel wrapper backed by our WebSocket policy server."""

    def __init__(self, host: str = "127.0.0.1", port: int = 20080,
                 action_chunk_size: int = CALVIN_ACTION_HORIZON):
        self.client = WebsocketClientPolicy(host=host, port=port)
        self.action_chunk_size = action_chunk_size
        self._pending_actions: "deque[np.ndarray]" = deque()
        self._current_goal: Optional[str] = None

    def reset(self):
        """Called at the beginning of every new subtask (language instruction)."""
        self._pending_actions.clear()
        self._current_goal = None

    def step(self, obs: dict, goal: str) -> np.ndarray:
        """One env step. Returns a 7-D relative-action vector."""
        # New subtask? clear the chunk buffer.
        if goal != self._current_goal:
            self._pending_actions.clear()
            self._current_goal = goal

        if not self._pending_actions:
            # Query the policy server for a fresh action chunk.
            static = _to_rgb256(obs["rgb_obs"]["rgb_static"])
            gripper = _to_rgb256(obs["rgb_obs"]["rgb_gripper"])
            state8 = _robot_obs_15d_to_8d(obs["robot_obs"])
            req = {
                "images": [[static, gripper]],       # [B=1, V=2] uint8 arrays
                "instructions": [str(goal)],
                "state": [state8],                    # [B, 8]
            }
            resp = self.client.infer(req)
            # server returns {"data": {"normalized_actions": [B, T, 7]}}
            actions = np.asarray(resp["data"]["normalized_actions"], dtype=np.float32)
            actions = actions[0]                      # [T, 7]
            # Only consume the first `action_chunk_size` predicted actions.
            for a in actions[: self.action_chunk_size]:
                self._pending_actions.append(a)

        return self._pending_actions.popleft()
