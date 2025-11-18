"""Model + controller implementations for all experimental conditions."""

from __future__ import annotations

import math
import random
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM

###############################################################################
# Helper: locate transformer blocks                                            #
###############################################################################

def _get_transformer_blocks(model: torch.nn.Module) -> List[torch.nn.Module]:
    names = {
        "GPTNeoXLayer",
        "QwenBlock",
        "GPT2Block",
        "TransformerBlock",
    }
    return [m for m in model.modules() if m.__class__.__name__ in names]

###############################################################################
# BLADE Controller                                                              
###############################################################################

class BladeController:
    """Layer-wise 1-bit gradient alignment LR scaler as described in the paper."""

    def __init__(self, blocks: List[torch.nn.Module], optimizer: torch.optim.Optimizer, cfg):
        self.enabled = bool(getattr(cfg, "blade", {}).get("enabled", False))
        if not self.enabled:
            return
        self.blocks = blocks
        self.opt = optimizer
        blade_cfg = cfg.blade
        self.K = int(blade_cfg.k_bits)
        self.beta = float(blade_cfg.beta_ema)
        self.min_c = float(blade_cfg.clamp_min)
        self.max_c = float(blade_cfg.clamp_max)
        self.refresh = int(blade_cfg.reservoir_refresh_interval)
        self.enable_reservoir = bool(blade_cfg.enable_reservoir)
        self.rng = random.Random(int(blade_cfg.indices_seed))

        # Pre-sample indices ----------------------------------------------------
        self.samples: List[List[Tuple[torch.Tensor, int]]] = []
        for blk in blocks:
            params = [p for p in blk.parameters() if p.requires_grad]
            layer_samples = []
            for _ in range(self.K):
                p = params[self.rng.randrange(len(params))]
                idx = self.rng.randrange(p.numel())
                layer_samples.append((p, idx))
            self.samples.append(layer_samples)

        self.prev_sign: List[int] = [0 for _ in blocks]
        self.ema_L = 0.0
        self.ema_H = 0.0
        self.ema_G = [0.0 for _ in blocks]
        self.ema_A = [1.0 for _ in blocks]
        self.step = 0

    @staticmethod
    def _popcnt(x: int) -> int:  # pragma: no cover
        return x.bit_count() if hasattr(int, "bit_count") else bin(x).count("1")

    def update(self, loss_val: float, entropy_val: float, base_lr: float):
        if not self.enabled:
            return
        self.step += 1

        # ---------------- Global statistics -----------------------------------
        self.ema_L = self.beta * self.ema_L + (1 - self.beta) * loss_val
        self.ema_H = self.beta * self.ema_H + (1 - self.beta) * entropy_val
        L_hat = self.ema_L / (1 - self.beta ** self.step)
        H_hat = self.ema_H / (1 - self.beta ** self.step)

        d_layer = []
        for li, blk in enumerate(self.blocks):
            # Gather gradient signs for K sampled parameters
            g_vals = []
            packed = 0
            for bi, (param, flat_idx) in enumerate(self.samples[li]):
                grad_flat = param.grad.view(-1)[flat_idx]
                g_val = grad_flat.item()
                g_vals.append(g_val)
                if g_val >= 0:
                    packed |= 1 << bi
            xor = packed ^ self.prev_sign[li]
            ham = self._popcnt(xor)
            a_t = 1 - ham / self.K
            self.prev_sign[li] = packed
            g_norm = math.sqrt(sum(g * g for g in g_vals))

            # EMA updates ------------------------------------------------------
            self.ema_G[li] = self.beta * self.ema_G[li] + (1 - self.beta) * g_norm
            self.ema_A[li] = self.beta * self.ema_A[li] + (1 - self.beta) * a_t
            G_hat = self.ema_G[li] / (1 - self.beta ** self.step)
            A_hat = self.ema_A[li] / (1 - self.beta ** self.step)

            d_main = (
                (loss_val / L_hat)
                * (entropy_val / H_hat)
                * (g_norm / max(1e-8, G_hat))
            ) ** (1 / 3)
            d_align = ((1 + a_t) / (1 + A_hat)) ** (-0.5)
            d = max(self.min_c, min(self.max_c, d_main * d_align))
            d_layer.append(d)

        d_bar = float(torch.median(torch.tensor(d_layer)))

        # Update LRs -----------------------------------------------------------
        for pg, d in zip(self.opt.param_groups[: len(self.blocks)], d_layer):
            base = pg.get("base_lr", base_lr)
            pg["lr"] = base * d_bar * d
        for pg in self.opt.param_groups[len(self.blocks) :]:
            base = pg.get("base_lr", base_lr)
            pg["lr"] = base * d_bar

        # Reservoir refresh ----------------------------------------------------
        if self.enable_reservoir and self.step % self.refresh == 0:
            for li, blk in enumerate(self.blocks):
                params = [p for p in blk.parameters() if p.requires_grad]
                for _ in range(2):
                    repl_pos = self.rng.randrange(self.K)
                    param = params[self.rng.randrange(len(params))]
                    idx = self.rng.randrange(param.numel())
                    self.samples[li][repl_pos] = (param, idx)

###############################################################################
# Sketch-Align (baseline)                                                      #
###############################################################################

class SketchAlignController:
    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, cfg):
        self.enabled = bool(getattr(cfg, "sketch_align", {}).get("enabled", False))
        if not self.enabled:
            return
        sa_cfg = cfg.sketch_align
        self.opt = optimizer
        self.K = int(sa_cfg.sketch_size)
        self.beta = float(sa_cfg.beta_ema)
        self.min_c = float(sa_cfg.clamp_min)
        self.max_c = float(sa_cfg.clamp_max)

        params = [p for p in model.parameters() if p.requires_grad]
        self.samples: List[Tuple[torch.Tensor, int]] = []
        rng = random.Random(0)
        for _ in range(self.K):
            p = params[rng.randrange(len(params))]
            idx = rng.randrange(p.numel())
            self.samples.append((p, idx))

        self.prev_sign = 0
        self.ema_L = 0.0
        self.ema_H = 0.0
        self.ema_G = 0.0
        self.ema_A = 1.0
        self.step = 0

    def _popcnt(self, x: int) -> int:  # pragma: no cover
        return x.bit_count() if hasattr(int, "bit_count") else bin(x).count("1")

    def update(self, loss_val: float, entropy_val: float):
        if not self.enabled:
            return
        self.step += 1
        g_vals = []
        packed = 0
        for bi, (param, idx) in enumerate(self.samples):
            gv = param.grad.view(-1)[idx].item()
            g_vals.append(gv)
            if gv >= 0:
                packed |= 1 << bi
        xor = packed ^ self.prev_sign
        ham = self._popcnt(xor)
        a_t = 1 - ham / self.K
        self.prev_sign = packed

        g_norm = math.sqrt(sum(g * g for g in g_vals))

        # EMA updates ---------------------------------------------------------
        self.ema_L = self.beta * self.ema_L + (1 - self.beta) * loss_val
        self.ema_H = self.beta * self.ema_H + (1 - self.beta) * entropy_val
        self.ema_G = self.beta * self.ema_G + (1 - self.beta) * g_norm
        self.ema_A = self.beta * self.ema_A + (1 - self.beta) * a_t

        L_hat = self.ema_L / (1 - self.beta ** self.step)
        H_hat = self.ema_H / (1 - self.beta ** self.step)
        G_hat = self.ema_G / (1 - self.beta ** self.step)
        A_hat = self.ema_A / (1 - self.beta ** self.step)

        d_main = ((loss_val / L_hat) * (entropy_val / H_hat) * (g_norm / G_hat)) ** (
            1 / 3
        )
        d_align = ((1 + a_t) / (1 + A_hat)) ** (-0.5)
        d = max(self.min_c, min(self.max_c, d_main * d_align))

        for pg in self.opt.param_groups:
            base_lr = pg.get("base_lr", pg["lr"])
            pg["lr"] = base_lr * d

###############################################################################
# Model + optimiser builder                                                    #
###############################################################################

def build_model_and_optimizer(cfg, device):
    torch_dtype = torch.float16 if str(cfg.model.dtype).lower() == "fp16" else torch.float32
    # Build kwargs for model loading
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "cache_dir": ".cache/",
    }
    # Only add device_map if CUDA is available (requires accelerate library)
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        **model_kwargs,
    )
    if cfg.model.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.to(device)

    blocks = _get_transformer_blocks(model)

    # Param groups – one per transformer block for fine-grained LR control -----
    param_groups: List[Dict] = []
    for blk in blocks:
        params = [p for p in blk.parameters() if p.requires_grad]
        param_groups.append({"params": params})

    seen_ids = {id(p) for pg in param_groups for p in pg["params"]}
    residual_params = [p for p in model.parameters() if p.requires_grad and id(p) not in seen_ids]
    if residual_params:
        param_groups.append({"params": residual_params})

    opt_cls = torch.optim.AdamW if cfg.training.optimizer.name.lower() == "adamw" else torch.optim.Adam
    optimizer = opt_cls(
        param_groups,
        lr=float(cfg.training.lr_scheduler.peak_lr),
        betas=tuple(cfg.training.optimizer.betas),
        eps=float(cfg.training.optimizer.eps),
        weight_decay=float(cfg.training.optimizer.weight_decay),
    )

    # Store base_lr for controllers -------------------------------------------
    for pg in optimizer.param_groups:
        pg["base_lr"] = cfg.training.lr_scheduler.peak_lr

    return model, optimizer, blocks
