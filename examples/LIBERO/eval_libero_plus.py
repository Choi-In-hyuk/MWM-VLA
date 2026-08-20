"""LIBERO-plus robustness eval for VLA-DINO-Mamba.

LIBERO-plus의 7축 perturbation(libero_10 = 2,519 변형)에서 모델을 평가.
- 각 task 변형 1 trial (논문 표준), task_classification.json의 category로 7축 집계.
- 결과는 per-task JSONL로 저장 -> 중단/재개 가능 + category별 성공률 집계.
- video는 성공/실패 각 N개까지만 (디스크 절약).

기존 eval_libero.py의 inference 흐름(M1Inference + env)을 그대로 재사용.
실행은 server_policy(모델 서버) + 이 스크립트 형태 (eval_libero_dino.sh와 동일 패턴).
"""
import collections
import dataclasses
import json
import logging
import os
import pathlib

import imageio
import numpy as np
import tqdm
import tyro

from libero.libero import benchmark, get_libero_path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# PyTorch 2.6+ : weights_only=True 기본값이라 plus의 init_states(.pt, numpy 포함) 로딩 실패.
# plus 코드 수정 없이 torch.load 기본값을 weights_only=False로 강제 (신뢰된 plus 자산).
import torch  # noqa: E402

_orig_torch_load = torch.load


def _torch_load_compat(*a, **kw):
    kw.setdefault("weights_only", False)
    return _orig_torch_load(*a, **kw)


torch.load = _torch_load_compat

# eval_libero.py의 검증된 헬퍼/상수 재사용 (중복 구현 금지)
from examples.LIBERO.eval_libero import (
    LIBERO_DUMMY_ACTION,
    LIBERO_ENV_RESOLUTION,
    _binarize_gripper_open,
    _quat2axisangle,
    short_name,
)
from examples.LIBERO.model2libero_interface import M1Inference

from libero.libero.envs import OffScreenRenderEnv


def _get_libero_env_plus(task, resolution, seed):
    """plus 호환 env 생성. plus env_wrapper는 bddl_file_name을 str로 파싱(`_view_` in ...)
    하므로 Path가 아닌 str 경로를 넘긴다 (eval_libero.py의 _get_libero_env는 Path를 넘겨 깨짐)."""
    task_description = task.language
    task_bddl_file = str(
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env, task_description

logging.basicConfig(level=logging.INFO)

# libero_10 데모 최장 505 steps (eval_libero.py와 동일)
MAX_STEPS = 520

# LIBERO-plus 7축 perturbation. benchmark 는 suite 를 category(축) 단위로 로드하므로
# (Benchmark.__init__(category_value=...)), 전체 평가는 이 7축을 순회하며 합산한다.
PLUS_CATEGORIES = [
    "Background Textures",
    "Camera Viewpoints",
    "Language Instructions",
    "Light Conditions",
    "Objects Layout",
    "Robot Initial States",
    "Sensor Noise",
]


@dataclasses.dataclass
class Args:
    pretrained_path: str = ""
    host: str = "127.0.0.1"
    port: int = 10093
    resize_size = [224, 224]

    task_suite_name: str = "libero_10"  # plus: libero_spatial/object/goal/10
    num_steps_wait: int = 10
    num_trials_per_task: int = 1  # 논문 표준: 변형당 1회

    out_dir: str = "results/eval_plus/libero_10_A_a"
    video_per_class: int = 10  # category당 성공/실패 영상 각 N개까지만
    max_tasks_per_category: int = 0  # 0=전체, >0=축당 N개만 (파일럿/시간측정용)
    only_category: str = ""  # 지정 시 이 category(축) 하나만 평가 (러너가 축 순서 제어)
    seed: int = 7

    with_state: str = "true"
    action_chunk_size: int = 0

    # task_classification.json (category 매핑). plus repo 내부.
    classification_json: str = (
        "/home/choi/LIBERO-plus/libero/libero/benchmark/task_classification.json"
    )


def load_category_map(path, suite):
    """task_classification.json -> {task_index: (category, difficulty)}.

    benchmark가 로드하는 task 순서(task_order 0..N-1)가 classification 리스트 순서와
    동일하다는 것을 import 검증에서 확인함 (n_tasks == len(classification[suite])).
    """
    d = json.load(open(path))
    entries = d[suite]
    cmap = {}
    for i, e in enumerate(entries):
        cmap[i] = (e["category"], e.get("difficulty_level", -1), e["name"])
    return cmap


def eval_plus(args: Args) -> None:
    logging.info(f"Args: {json.dumps(dataclasses.asdict(args), indent=2)}")
    np.random.seed(args.seed)

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    vid_dir = out / "videos"
    vid_dir.mkdir(exist_ok=True)
    results_path = out / "results.jsonl"

    # resume: 이미 평가한 task_index 수집
    done_ids = set()
    if results_path.exists():
        with open(results_path) as fh:
            for line in fh:
                try:
                    done_ids.add(json.loads(line)["task_index"])
                except Exception:
                    pass
    logging.info(f"resume: {len(done_ids)} tasks already evaluated")

    # LIBERO-plus 리포는 suite 하나를 "한 category(축)"씩 로드한다
    # (Benchmark.__init__(category_value=...)). 7축 전체를 돌려면 category마다
    # task_suite 를 다시 만들어 순회해야 한다. task_index 는 (category, local_id)
    # 조합으로 유일하게 식별한다 (resume 도 이 조합 기준).
    benchmark_dict = benchmark.get_benchmark_dict()

    model = M1Inference(
        policy_ckpt_path=args.pretrained_path,
        host=args.host,
        port=args.port,
        image_size=args.resize_size,
        action_chunk_size=args.action_chunk_size,
    )

    # category별 영상 저장 카운터
    vid_count = collections.defaultdict(lambda: {"success": 0, "failure": 0})

    # classification.json: {suite: [{id, category, name, ...}]}. id 는 1-based 전역
    # 인덱스인데 LIBERO-plus benchmark._make_benchmark 는 이 id 를 0-based task_maps
    # 리스트에 그대로 인덱싱한다(off-by-one -> id_max 축에서 IndexError). 여기서
    # id-1 로 보정한 task 리스트를 직접 만들어 task_suite.tasks 를 덮어써 우회한다.
    import libero.libero.benchmark as _B
    cls_all = json.load(open(args.classification_json))[args.task_suite_name]
    all_tasks = list(_B.task_maps[args.task_suite_name].values())
    cat_to_entries = collections.defaultdict(list)
    for e in cls_all:
        cat_to_entries[e["category"]].append(e)

    categories = PLUS_CATEGORIES
    if args.only_category:
        assert args.only_category in PLUS_CATEGORIES, (
            f"unknown category {args.only_category!r}; valid={PLUS_CATEGORIES}"
        )
        categories = [args.only_category]

    fout = open(results_path, "a")
    for category in tqdm.tqdm(categories, desc="category"):
        entries = cat_to_entries.get(category, [])
        # -1 보정한 올바른 task 리스트 (id 순서 유지)
        cat_tasks = [all_tasks[e["id"] - 1] for e in entries]
        # 생성자 IndexError 회피: 안전 축으로 만든 뒤 tasks 를 교체
        task_suite = benchmark_dict[args.task_suite_name](category_value="Camera Viewpoints")
        task_suite.tasks = cat_tasks
        task_suite.n_tasks = len(cat_tasks)
        n_cat = task_suite.n_tasks
        logging.info(f"[{category}] {n_cat} tasks")

        # 파일럿: 축당 N개만
        if args.max_tasks_per_category > 0:
            local_ids = list(range(min(args.max_tasks_per_category, n_cat)))
        else:
            local_ids = list(range(n_cat))

        for local_id in tqdm.tqdm(local_ids, desc=category, leave=False):
            task_index = f"{category}#{local_id}"   # 유일 식별자 (resume 기준)
            if task_index in done_ids:
                continue
            task_names = task_suite.get_task_names()
            cname = task_names[local_id] if local_id < len(task_names) else str(local_id)
            difficulty = -1
            task = task_suite.get_task(local_id)
            initial_states = task_suite.get_task_init_states(local_id)
            env, task_description = _get_libero_env_plus(task, LIBERO_ENV_RESOLUTION, args.seed)

            for episode_idx in range(args.num_trials_per_task):
                model.reset(task_description=task_description)
                env.reset()
                env.set_init_state(initial_states[episode_idx % len(initial_states)])

                t, step, done = 0, 0, False
                replay_images = []
                while t < MAX_STEPS + args.num_steps_wait:
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(
                        obs["robot0_eye_in_hand_image"][::-1, ::-1]
                    )
                    replay_images.append(img)

                    state = np.concatenate(
                        (
                            obs["robot0_eef_pos"],
                            _quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        )
                    )
                    obs_input = {
                        "images": [img, wrist_img],
                        "task_description": str(task_description),
                        "step": step,
                    }
                    if args.with_state == "true":
                        obs_input["state"] = np.expand_dims(state, axis=0)

                    response = model.step(**obs_input)
                    raw = response["raw_action"]
                    wv = np.asarray(raw.get("world_vector"), dtype=np.float32).reshape(-1)
                    rot = np.asarray(raw.get("rotation_delta"), dtype=np.float32).reshape(-1)
                    grip = _binarize_gripper_open(
                        np.asarray(raw.get("open_gripper"), dtype=np.float32).reshape(-1)
                    )
                    delta_action = np.concatenate([wv, rot, grip], axis=0)

                    obs, reward, done, info = env.step(delta_action.tolist())
                    if done:
                        break
                    t += 1
                    step += 1

                # 결과 기록
                rec = {
                    "task_index": task_index,
                    "category": category,
                    "difficulty": difficulty,
                    "name": cname,
                    "task": str(task_description),
                    "success": bool(done),
                }
                fout.write(json.dumps(rec) + "\n")
                fout.flush()

                # 영상: category당 성공/실패 각 N개까지만
                key = "success" if done else "failure"
                if vid_count[category][key] < args.video_per_class:
                    vid_count[category][key] += 1
                    safe_cat = category.replace(" ", "_")
                    imageio.mimwrite(
                        vid_dir / f"{safe_cat}_{key}_{short_name(str(task_description))}.mp4",
                        [np.asarray(x) for x in replay_images],
                        fps=10,
                    )

            env.close()

    fout.close()
    summarize(results_path)


def summarize(results_path):
    """category별 성공률 집계 + 전체."""
    from collections import defaultdict

    by = defaultdict(lambda: [0, 0])  # category -> [success, total]
    tot = [0, 0]
    with open(results_path) as fh:
        for line in fh:
            r = json.loads(line)
            c = r["category"]
            by[c][1] += 1
            tot[1] += 1
            if r["success"]:
                by[c][0] += 1
                tot[0] += 1

    print("\n=========== LIBERO-plus SUMMARY ===========")
    print(f"{'category':<26} {'succ/total':>12} {'rate':>8}")
    for c in sorted(by):
        s, n = by[c]
        print(f"{c:<26} {f'{s}/{n}':>12} {s/max(n,1)*100:>7.1f}%")
    print(f"{'-- TOTAL --':<26} {f'{tot[0]}/{tot[1]}':>12} {tot[0]/max(tot[1],1)*100:>7.1f}%")


if __name__ == "__main__":
    tyro.cli(eval_plus)
