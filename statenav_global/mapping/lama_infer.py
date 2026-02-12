# lama_infer.py
# Minimal LaMa (saicinpainting) generator loader for IN-MEMORY inference.
# Avoids importing training.trainers / datasets (albumentations/imgaug/skimage),
# so it won't trip the NumPy/skimage incompat you saw.

from __future__ import annotations

import os
import sys
from typing import Tuple, List, Dict, Any, Optional

import torch
from omegaconf import OmegaConf

# Add the 'lama' directory to the Python path so 'saicinpainting' is directly importable
_current_dir = os.path.dirname(os.path.abspath(__file__))
_lama_dir = os.path.join(_current_dir, 'lama')
if _lama_dir not in sys.path:
    sys.path.insert(0, _lama_dir)

from saicinpainting.training.modules import make_generator


def _load_ckpt_obj(ckpt_path: str, device: str) -> Any:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    return torch.load(ckpt_path, map_location=device)


def _get_raw_state_dict(ckpt_obj: Any) -> Dict[str, torch.Tensor]:
    """
    Handle common checkpoint formats.
    In your case, best.ckpt is already a flat state_dict (keys like 'generator.model....').
    But we also support Lightning-style ckpts with a 'state_dict' field.
    """
    if isinstance(ckpt_obj, dict) and "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
        return ckpt_obj["state_dict"]
    if isinstance(ckpt_obj, dict) and all(isinstance(k, str) for k in ckpt_obj.keys()):
        # assume it's already a state_dict
        return ckpt_obj  # type: ignore[return-value]
    raise RuntimeError(
        "Unrecognized checkpoint format. Expected dict with 'state_dict' or a flat state_dict dict."
    )


def _extract_generator_state_dict(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Your checkpoint keys look like:
      'generator.model.1.ffc.convl2l.weight', ...
    Your instantiated generator's state_dict keys typically look like:
      'model.1.ffc.convl2l.weight', ...
    So we STRIP ONLY the leading 'generator.' prefix.
    """
    gen_sd: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if k.startswith("generator."):
            gen_sd[k[len("generator."):]] = v
        # else: ignore anything not belonging to generator
    if not gen_sd:
        # fall back: maybe checkpoint already matches generator keys without 'generator.'
        # (rare, but safe)
        gen_sd = sd
    return gen_sd


def _build_generator_from_cfg(cfg_path: str, device: str) -> torch.nn.Module:
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    cfg = OmegaConf.load(cfg_path)

    # Your config.yaml has top-level 'generator:' section.
    if "generator" not in cfg:
        raise RuntimeError(f"No top-level 'generator' key found in {cfg_path}. Keys: {list(cfg.keys())}")

    gen_cfg = cfg.generator
    gen_kwargs = OmegaConf.to_container(gen_cfg, resolve=True)

    if not isinstance(gen_kwargs, dict):
        raise RuntimeError("generator config did not resolve to a dict")

    kind = gen_kwargs.pop("kind", "ffc_resnet")

    # IMPORTANT: This repo's make_generator expects constructor args via **kwargs.
    # Passing gen_cfg alone is not enough.
    gen = make_generator(gen_cfg, kind=kind, **gen_kwargs).to(device)
    gen.eval()
    return gen


def load_lama_generator(
    cfg_path: str,
    ckpt_path: str,
    device: str = "cuda",
    strict: bool = False,
    verbose: bool = True,
) -> Tuple[torch.nn.Module, List[str], List[str]]:
    """
    Returns:
      generator (torch.nn.Module in eval mode),
      missing_keys (list),
      unexpected_keys (list)

    Usage:
      gen, missing, unexpected = load_lama_generator(".../big-lama/config.yaml", ".../big-lama/models/best.ckpt")
    """
    gen = _build_generator_from_cfg(cfg_path, device)

    ckpt_obj = _load_ckpt_obj(ckpt_path, device=device)
    sd = _get_raw_state_dict(ckpt_obj)
    gen_sd = _extract_generator_state_dict(sd)

    # Load weights
    incompatible = gen.load_state_dict(gen_sd, strict=strict)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)

    if verbose:
        print(f"[LaMa] Loaded generator from:\n  cfg:  {cfg_path}\n  ckpt: {ckpt_path}")
        print(f"[LaMa] strict={strict}  missing={len(missing)}  unexpected={len(unexpected)}")
        if len(missing) > 0:
            print("[LaMa] missing sample:", missing[:20])
        if len(unexpected) > 0:
            print("[LaMa] unexpected sample:", unexpected[:20])

    gen.eval()
    return gen, missing, unexpected