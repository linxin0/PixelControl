"""Standalone three-control validation/inference entry.

Given a trained PixelDiT three-control checkpoint, generate the 7 validation
modes (depth / seg / edge / depth_seg / depth_edge / seg_edge /
depth_seg_edge) for up to 500 images, then optionally run the PixelGen metric
scripts requested by the user.

Example:
  CUDA_VISIBLE_DEVICES=0 python t2i/infer_threecontrol_val.py \
    --checkpoint t2i/universal_pix_t2i_workdirs/exp_pixeldit_threecontrol_v1_from_depth10k/checkpoints/epoch_1_step_2000.pth \
    --output_dir t2i/universal_pix_t2i_workdirs/exp_pixeldit_threecontrol_v1_from_depth10k/val/iter_2000_eval500 \
    --max_samples 500 \
    --run_metrics
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
import re
import subprocess
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_T2I_ROOT = Path(__file__).resolve().parent
if str(_T2I_ROOT) not in sys.path:
    sys.path.insert(0, str(_T2I_ROOT))

from diffusion.model.builder import build_model, get_tokenizer_and_text_encoder  # noqa: E402
from diffusion.utils.config import DataConfig, ModelConfig, AEConfig, TextEncoderConfig, SchedulerConfig, TrainingConfig  # noqa: E402
from diffusion.utils.config import model_init_config  # noqa: E402

try:
    import pyrallis as _pyrallis  # noqa: F401
except Exception:
    # ``train_control.py`` is imported for its validation helper and config
    # dataclasses. It decorates ``main`` with ``@pyrallis.wrap()``, so provide
    # a no-op fallback when running this standalone infer script from a lighter
    # environment. In the normal training env, the real pyrallis is used.
    sys.modules["pyrallis"] = types.SimpleNamespace(wrap=lambda *a, **k: (lambda fn: fn))

from train_control import ControlConfig, CycleLossConfig, PixDiTControlConfig, ValidationConfig, run_control_validation  # noqa: E402

# Side-effect registrations.
import diffusion.model.control_trainer  # noqa: E402,F401
import diffusion.data.datasets.control_datasets  # noqa: E402,F401


@dataclass
class Args:
    config_path: str = "t2i/configs_t2i/pixeldit_threecontrol_v1.yaml"
    checkpoint: str = ""
    output_dir: str = ""
    max_samples: int = 500
    batch_size: int = 8
    num_sampling_steps: int = 50
    cfg_scale: float = 2.75
    seed: int = 2025
    device: str = "cuda:0"
    dtype: str = "bf16"
    load_ema: bool = False
    run_metrics: bool = False
    pixelgen_root: str = "./third_party/PixelGen"
    clip_model: str = "./pretrained/clip-vit-large-patch14"
    metrics_batch_size: int = 16
    metrics_device: str = "cuda:0"
    output_metrics_json: str = ""
    output_depth_json: str = ""
    control_modes: str = ""


def _abs_path(path: str) -> str:
    if not path:
        return path
    if osp.isabs(path):
        return path
    return osp.abspath(path)


def _infer_step_from_ckpt(ckpt: str) -> str:
    m = re.search(r"_step_(\d+)", osp.basename(ckpt))
    if m:
        return m.group(1)
    return "custom"


def _torch_dtype(dtype: str) -> torch.dtype:
    dtype = str(dtype).lower()
    if dtype in ("bf16", "bfloat16"):
        return torch.bfloat16
    if dtype in ("fp16", "float16", "half"):
        return torch.float16
    if dtype in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def _load_config(config_path: str) -> PixDiTControlConfig:
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    def _filter(cls, data):
        names = set(cls.__dataclass_fields__.keys())
        return {k: v for k, v in dict(data or {}).items() if k in names}

    cfg = PixDiTControlConfig()
    cfg.data = DataConfig(**_filter(DataConfig, raw.get("data", {})))
    cfg.model = ModelConfig(**_filter(ModelConfig, raw.get("model", {})))
    cfg.vae = AEConfig(**_filter(AEConfig, raw.get("vae", {})))
    cfg.text_encoder = TextEncoderConfig(**_filter(TextEncoderConfig, raw.get("text_encoder", {})))
    cfg.scheduler = SchedulerConfig(**_filter(SchedulerConfig, raw.get("scheduler", {})))
    cfg.train = TrainingConfig(**_filter(TrainingConfig, raw.get("train", {})))
    cfg.validation = ValidationConfig(**_filter(ValidationConfig, raw.get("validation", {})))

    control_raw = dict(raw.get("control", {}) or {})
    cycle_raw = control_raw.get("cycle_loss", None)
    control_raw = _filter(ControlConfig, control_raw)
    if isinstance(cycle_raw, dict):
        control_raw["cycle_loss"] = CycleLossConfig(
            type=cycle_raw.get("type"),
            init_args=cycle_raw.get("init_args", {}) or {},
        )
    cfg.control = ControlConfig(**control_raw)

    for k in ("work_dir", "report_to", "tracker_project_name", "name", "loss_report_name", "resume_from", "load_from", "debug", "caching"):
        if k in raw:
            setattr(cfg, k, raw[k])
    return cfg


def run_metrics(args: Args, gen_dir: str) -> None:
    pixelgen_root = _abs_path(args.pixelgen_root)
    os.makedirs(osp.join(pixelgen_root, "outputs"), exist_ok=True)
    step = _infer_step_from_ckpt(args.checkpoint)
    metrics_json = args.output_metrics_json or osp.join(
        pixelgen_root, "outputs", f"pixeldit_threecontrol_eval_metrics_iter{step}.json"
    )
    depth_json = args.output_depth_json or osp.join(
        pixelgen_root, "outputs", f"pixeldit_threecontrol_depth_consistency_iter{step}.json"
    )
    env = os.environ.copy()
    env_device = str(args.metrics_device)
    if env_device.startswith("cuda:"):
        env["CUDA_VISIBLE_DEVICES"] = env_device.split(":", 1)[1]
        metric_device_arg = "cuda:0"
    else:
        metric_device_arg = env_device

    cmd_metrics = [
        sys.executable,
        osp.join(pixelgen_root, "scripts/eval_depth_metrics.py"),
        "--gen_dirs",
        gen_dir,
        "--output_json",
        metrics_json,
        "--metrics",
        "fid",
        "clip_text",
        "clip_img",
        "lpips",
        "--clip_model",
        args.clip_model,
        "--batch_size",
        str(args.metrics_batch_size),
        "--device",
        metric_device_arg,
    ]
    print("[infer_threecontrol_val] running:", " ".join(cmd_metrics), flush=True)
    subprocess.run(cmd_metrics, check=True, env=env)

    cmd_depth = [
        sys.executable,
        osp.join(pixelgen_root, "scripts/eval_depth_consistency_da3.py"),
        "--gen_dirs",
        gen_dir,
        "--output_json",
        depth_json,
    ]
    print("[infer_threecontrol_val] running:", " ".join(cmd_depth), flush=True)
    subprocess.run(cmd_depth, check=True, env=env)
    print(f"[infer_threecontrol_val] metrics json: {metrics_json}", flush=True)
    print(f"[infer_threecontrol_val] depth consistency json: {depth_json}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    for field_name, field_def in Args.__dataclass_fields__.items():
        default = field_def.default
        arg_type = type(default)
        if arg_type is bool:
            parser.add_argument(f"--{field_name}", action="store_true")
        else:
            parser.add_argument(f"--{field_name}", type=arg_type, default=default)
    ns = parser.parse_args()
    args = Args(**vars(ns))
    if not args.checkpoint:
        raise ValueError("--checkpoint is required")

    config_path = _abs_path(args.config_path)
    ckpt_path = _abs_path(args.checkpoint)
    cfg = _load_config(config_path)
    cfg.validation.max_samples = int(args.max_samples)
    cfg.validation.batch_size = int(args.batch_size)
    cfg.validation.num_sampling_steps = int(args.num_sampling_steps)
    cfg.validation.cfg_scale = float(args.cfg_scale)
    cfg.validation.seed = int(args.seed)
    if args.control_modes:
        cfg.validation.control_modes = [m.strip() for m in args.control_modes.split(",") if m.strip()]

    step = _infer_step_from_ckpt(ckpt_path)
    if args.output_dir:
        output_dir = _abs_path(args.output_dir)
    else:
        output_dir = osp.abspath(
            osp.join(_T2I_ROOT, cfg.work_dir, "val", f"iter_{step}_eval{args.max_samples}")
        )
    work_dir = osp.dirname(osp.dirname(output_dir))
    cfg.validation.save_dir = osp.basename(osp.dirname(output_dir))
    global_step_name = osp.basename(output_dir)
    if global_step_name.startswith("iter_"):
        global_step = global_step_name[len("iter_"):]
    else:
        global_step = global_step_name

    device = torch.device(args.device)
    dtype = _torch_dtype(args.dtype)
    torch.backends.cuda.matmul.allow_tf32 = True

    tokenizer, text_encoder = get_tokenizer_and_text_encoder(
        name=cfg.text_encoder.text_encoder_name,
        device=device,
    )
    text_encoder.eval()

    model_kwargs = model_init_config(cfg, latent_size=cfg.model.image_size)
    model = build_model(
        cfg.model.model,
        False,
        getattr(cfg.model, "fp32_attention", False),
        null_embed_path="",
        **model_kwargs,
    ).to(device=device, dtype=dtype).eval()

    # Load model weights only; do not resume optimizer/scheduler.
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd_key = "state_dict_ema" if args.load_ema and "state_dict_ema" in ckpt else "state_dict"
    state_dict = ckpt.get(sd_key, ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(
        f"[infer_threecontrol_val] loaded {sd_key} from {ckpt_path}; "
        f"missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )

    # Build null prompt embedding for classifier-free guidance.
    with torch.no_grad():
        null_tokens = tokenizer(
            "",
            max_length=cfg.text_encoder.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(device)
        null_embed = text_encoder(null_tokens.input_ids, attention_mask=null_tokens.attention_mask)[0].detach()
        null_y = null_embed[:, None].to(dtype=dtype)
        null_y_mask = null_tokens.attention_mask[:, None, None, :]

    class _Logger:
        def info(self, msg):
            print(msg, flush=True)

        def warning(self, msg):
            print("WARNING:", msg, flush=True)

    run_control_validation(
        model=model,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        null_y=null_y,
        null_y_mask=null_y_mask,
        validation_cfg=cfg.validation,
        text_encoder_cfg=cfg.text_encoder,
        scheduler_cfg=cfg.scheduler,
        work_dir=work_dir,
        global_step=global_step,
        device=device,
        dtype=dtype,
        logger=_Logger(),
        rank=0,
        world_size=1,
    )
    print(f"[infer_threecontrol_val] generated: {output_dir}", flush=True)

    if args.run_metrics:
        run_metrics(args, output_dir)


if __name__ == "__main__":
    main()
