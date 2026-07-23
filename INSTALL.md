# Installation & Environment

This project was developed and tested on:

- **OS:** Ubuntu 22.04 (Linux 6.8)
- **GPU:** 2× NVIDIA RTX PRO 6000 Blackwell (48 GB VRAM each). Any modern CUDA GPU with ≥24 GB VRAM should work for single-GPU training; DeepSpeed / DDP is used for 2-GPU runs.
- **NVIDIA driver:** 590.x
- **CUDA toolkit:** 13.0 (PyTorch is built for `cu130`)
- **Python:** 3.10

If you are on a different CUDA/PyTorch pairing, replace the PyTorch install line below with the matching wheel from https://pytorch.org.

---

## 1. Create the conda environment

The training / inference environment is called **`vla_jepa`**. Two equivalent ways:

### Option A — from `environment/vla_jepa.yml` (recommended)

```bash
conda env create -f environment/vla_jepa.yml
conda activate vla_jepa
```

### Option B — manual (Python 3.10 + PyTorch + pip requirements)

```bash
conda create -n vla_jepa python=3.10 -y
conda activate vla_jepa

# PyTorch 2.12 for CUDA 13.0 — adjust if your CUDA differs
pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cu130

pip install -r environment/vla_jepa_requirements.txt
```

---

## 2. Install `mamba-ssm`

`mamba-ssm` (used for the Mamba world model) needs a matching CUDA toolchain and is compiled on install. It usually fails silently on toolchain mismatch — check `import mamba_ssm` runs without errors.

```bash
pip install mamba-ssm==2.3.2.post1 causal-conv1d==1.5.2
python -c "import mamba_ssm; print('mamba_ssm OK')"
```

If installation fails, see https://github.com/state-spaces/mamba for platform-specific notes.

---

## 3. Install this repo (editable)

```bash
pip install -e .
```

This registers the `starVLA` package so training / eval scripts can import it.

---

## 4. Backbone checkpoints

Pretrained weights the codebase expects at load time:

- **DINOv2** (ViT-B/14) — loaded via `timm` on first run.
- **Qwen3-VL** — downloaded via `transformers` on first run (needs Hugging Face access to the Qwen3-VL repo).

MWM-VLA's own trained checkpoints (Streaming Mamba WM + Qwen-LoRA + flow-matching head) will be released separately — see the README for the "Model release" section.

---

## 5. Datasets

Datasets are kept **outside the repo** (see `.gitignore`). Set the paths in the eval / training scripts, or export env vars:

| Dataset | Purpose | Where the code looks |
|---|---|---|
| LIBERO (`libero_10`, `libero_all`) | Fine-tuning + eval | `LIBERO_HOME` env var in `scripts/eval_libero_*.sh` |
| LIBERO-Plus | Robustness eval (SOTA benchmark) | `liberoplus_config/config.yaml` |
| Something-Something v2 (SSv2) | Pretraining video source | Configured in `scripts/pretrain_ssv2_droid.py` |
| Droid | Pretraining video source | Configured in `scripts/pretrain_ssv2_droid.py` |

Follow the base VLA-JEPA repo instructions for LIBERO installation. LIBERO-Plus setup follows its own upstream benchmark; the config file in `liberoplus_config/` is the entry point.

---

## 6. Sanity check

```bash
conda activate vla_jepa
python -c "
import torch, mamba_ssm, transformers, starVLA
print('torch', torch.__version__, 'cuda?', torch.cuda.is_available(), torch.cuda.device_count(), 'GPUs')
print('mamba_ssm OK, transformers', transformers.__version__)
"
```

Expected output:

```
torch 2.12.0+cu130 cuda? True 2 GPUs
mamba_ssm OK, transformers 4.57.0
```

---

## Notes on other envs

The repo also references two evaluation-only conda envs from local development (`liberoplus_env`, `calvin_env`). These are **not required** to train or run MWM-VLA — they exist only to keep the LIBERO-Plus / CALVIN benchmark clients isolated from `mamba-ssm`'s CUDA build. Follow each benchmark's official install guide if you want to reproduce those eval numbers.
