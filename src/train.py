import copy
import json
import os
import random
import time
from typing import Any, Dict, List, Tuple

import hydra
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from src.preprocess import build_dataloaders, exact_match_metric
from src.model import (
    BladeController,
    SketchAlignController,
    build_model_and_optimizer,
)

# Optional import (trial-mode may disable WandB)
try:
    import wandb  # type: ignore
except ImportError:  # pragma: no cover
    wandb = None  # type: ignore

################################################################################
# Utilities                                                                    #
################################################################################

def seed_everything(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def autoregressive_ce_loss(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    answer_ids: List[torch.Tensor],
    pad_token_id: int,
) -> Tuple[torch.Tensor, float]:
    """Compute negative-log-likelihood of the answers without *ever* feeding the
    answer tokens to the network inputs (prevents label leakage).  We roll out
    the model autoregressively one token at a time, caching KV states so the
    cost is minimal.  Returns (loss, first-token entropy)."""

    device = input_ids.device
    B = input_ids.size(0)

    # Prompt forward pass ------------------------------------------------------
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    past = out.past_key_values
    first_logits = out.logits[:, -1, :]
    entropy_val = (
        torch.distributions.Categorical(logits=first_logits).entropy().mean().item()
    )

    total_nll = torch.tensor(0.0, device=device)
    total_tok = 0
    next_input_ids = None
    max_answer_len = max(ans.size(0) for ans in answer_ids)

    for t in range(max_answer_len):
        if t == 0:
            logits = first_logits  # avoid one forward at t=0
        else:
            logits = model(
                input_ids=next_input_ids, use_cache=True, past_key_values=past
            ).logits[:, -1, :]

        tgt = torch.full((B,), pad_token_id, dtype=torch.long, device=device)
        mask = torch.zeros(B, dtype=torch.bool, device=device)
        for b, ans in enumerate(answer_ids):
            if t < ans.size(0):
                tgt[b] = ans[t]
                mask[b] = True
        if mask.sum() == 0:
            break

        loss_t = F.cross_entropy(logits[mask], tgt[mask], reduction="sum")
        total_nll = total_nll + loss_t
        total_tok += int(mask.sum())

        pred_tokens = logits.argmax(-1)
        next_input_ids = pred_tokens.view(-1, 1)
        past = model(
            input_ids=next_input_ids, use_cache=True, past_key_values=past
        ).past_key_values

    mean_loss = total_nll / max(1, total_tok)
    return mean_loss, entropy_val


################################################################################
# Validation loop                                                              #
################################################################################

def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    tokenizer,
    device: torch.device,
    pad_id: int,
    max_batches: int = -1,
):
    model.eval()
    total_loss, total_tok = 0.0, 0
    correct, total = 0, 0
    preds_all, gold_all = [], []

    with torch.no_grad():
        for bi, batch in enumerate(dataloader):
            inp = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            ans_ids: List[torch.Tensor] = batch["answer_ids"]
            ans_txt: List[str] = batch["answer_text"]

            loss, _ = autoregressive_ce_loss(model, inp, attn, ans_ids, pad_id)
            total_loss += loss.item() * len(ans_ids)
            total_tok += len(ans_ids)

            # Generation for EM accuracy --------------------------------------
            gen_out = model.generate(
                input_ids=inp,
                attention_mask=attn,
                max_new_tokens=128,
                do_sample=False,
            )
            prompt_lens = attn.sum(-1).tolist()
            for b, ids in enumerate(gen_out):
                pred_ids = ids[prompt_lens[b] :]
                pred = tokenizer.decode(pred_ids, skip_special_tokens=True).strip()
                gold = ans_txt[b]
                preds_all.append(pred)
                gold_all.append(gold)
                if exact_match_metric(pred, gold):
                    correct += 1
                total += 1

            if 0 < max_batches <= bi:
                break

    model.train()
    return total_loss / max(1, total_tok), 100 * correct / max(1, total), preds_all, gold_all

################################################################################
# Optuna integration                                                           #
################################################################################

def _apply_cfg_path(cfg: Any, dotted_key: str, value: Any):
    """Given an OmegaConf object, set a *nested* key specified by dot-notation."""
    keys = dotted_key.split(".")
    ctx = cfg
    for k in keys[:-1]:
        if isinstance(ctx, dict):
            if k not in ctx:
                ctx[k] = {}
            ctx = ctx[k]
        else:
            if not hasattr(ctx, k):
                setattr(ctx, k, {})
            ctx = getattr(ctx, k)
    if isinstance(ctx, dict):
        ctx[keys[-1]] = value
    else:
        setattr(ctx, keys[-1], value)


def _suggest_param(trial, name: str, spec: Dict[str, Any]):
    p_type = spec["type"].lower()
    if p_type == "uniform":
        return trial.suggest_float(name, spec["low"], spec["high"])
    if p_type == "loguniform":
        return trial.suggest_float(name, spec["low"], spec["high"], log=True)
    if p_type == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    raise ValueError(f"Unknown Optuna parameter type: {p_type}")


def run_optuna(cfg) -> Dict[str, Any]:
    import optuna  # local import to avoid obligate dependency if not used

    # We optimise *validation exact-match* after a very small training budget ----
    def objective(trial):
        trial_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
        for dotted, spec in cfg.optuna.search_space.items():
            param_val = _suggest_param(trial, dotted, spec)
            _apply_cfg_path(trial_cfg, dotted, param_val)

        # Fast training run (few steps) ----------------------------------------
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_loader, val_loader, tokenizer = build_dataloaders(trial_cfg, ".cache/")
        pad_id = tokenizer.pad_token_id
        model, optimizer, blocks = build_model_and_optimizer(trial_cfg, device)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(trial_cfg.training.lr_scheduler.warmup_steps),
            num_training_steps=100,
        )

        blade_ctl = (
            BladeController(blocks, optimizer, trial_cfg)
            if trial_cfg.get("blade", {}).get("enabled", False)
            else None
        )
        sketch_ctl = (
            SketchAlignController(model, optimizer, trial_cfg)
            if trial_cfg.get("sketch_align", {}).get("enabled", False)
            else None
        )

        grad_accum = int(trial_cfg.training.gradient_accumulation_steps)
        scaler = torch.cuda.amp.GradScaler(enabled=bool(trial_cfg.training.fp16))
        global_step = 0
        model.train()
        for batch in train_loader:
            inp = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            ans_ids = batch["answer_ids"]
            with torch.cuda.amp.autocast(enabled=bool(trial_cfg.training.fp16)):
                loss, ent = autoregressive_ce_loss(model, inp, attn, ans_ids, pad_id)
                loss_scaled = loss / grad_accum
            scaler.scale(loss_scaled).backward()
            if (global_step + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), trial_cfg.training.max_grad_norm)
                if blade_ctl:
                    blade_ctl.update(loss.item(), ent, scheduler.get_last_lr()[0])
                if sketch_ctl:
                    sketch_ctl.update(loss.item(), ent)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            global_step += 1
            if global_step >= 100:
                break

        val_loss, val_acc, _, _ = evaluate(
            model,
            val_loader,
            tokenizer,
            device,
            pad_id,
            max_batches=2,
        )
        # Free GPU memory -------------------------------------------------------
        del model
        torch.cuda.empty_cache()
        return val_acc

    study = optuna.create_study(direction=cfg.optuna.direction)
    study.optimize(objective, n_trials=int(cfg.optuna.n_trials))
    return study.best_params

################################################################################
# Main training entry-point                                                    #
################################################################################

@hydra.main(config_path="../config", config_name="config")
def main(cfg):  # noqa: C901 – main loop is inevitably lengthy
    # ------------------------------------------------------------------
    # Resolve run-level settings
    # ------------------------------------------------------------------
    if not hasattr(cfg, "run_id") or cfg.run_id is None:
        raise ValueError("Config group 'run' missing. Launch with run=<run_id>.")

    run_id: str = cfg.run_id

    # Mode-dependent overrides ----------------------------------------
    if cfg.mode == "trial":
        cfg.wandb.mode = "disabled"
        cfg.optuna.n_trials = 0
        cfg.training.num_epochs = 1  # type: ignore[attr-defined]
        cfg.training.logging_steps = 1  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Hyper-parameter search (Optuna)                                   
    # ------------------------------------------------------------------
    best_params: Dict[str, Any] = {}
    if int(cfg.optuna.n_trials) > 0:
        best_params = run_optuna(cfg)
        for dotted_key, val in best_params.items():
            _apply_cfg_path(cfg, dotted_key, val)
        print("[Optuna] Best params:", json.dumps(best_params, indent=2))

    # ------------------------------------------------------------------
    # Preparation (seed, data, model)                                   
    # ------------------------------------------------------------------
    original_cwd = get_original_cwd()
    os.chdir(original_cwd)

    seed_everything(int(cfg.training.seed))  # type: ignore[attr-defined]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, tokenizer = build_dataloaders(cfg, ".cache/")
    pad_id = tokenizer.pad_token_id

    model, optimizer, blocks = build_model_and_optimizer(cfg, device)

    blade_ctl = (
        BladeController(blocks, optimizer, cfg) if getattr(cfg, "blade", {}).get("enabled", False) else None
    )
    sketch_ctl = (
        SketchAlignController(model, optimizer, cfg)
        if getattr(cfg, "sketch_align", {}).get("enabled", False)
        else None
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(cfg.training.lr_scheduler.warmup_steps),
        num_training_steps=len(train_loader) * int(cfg.training.num_epochs),  # type: ignore[attr-defined]
    )

    # ------------------------------------------------------------------
    # WandB initialisation                                              
    # ------------------------------------------------------------------
    wb = None
    if cfg.wandb.mode != "disabled" and wandb is not None:
        wb = wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            id=run_id,
            config=OmegaConf.to_container(cfg, resolve=True),
            resume="allow",
            mode=cfg.wandb.mode,
        )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    # If model is already in FP16, we don't need GradScaler (it's for mixed precision training)
    model_is_fp16 = str(cfg.model.dtype).lower() == "fp16"
    use_amp = bool(cfg.training.fp16) and not model_is_fp16
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)  # type: ignore[attr-defined]
    grad_accum = int(cfg.training.gradient_accumulation_steps)  # type: ignore[attr-defined]
    global_step = 0
    best_val_acc = -float("inf")
    best_epoch = -1

    for epoch in range(int(cfg.training.num_epochs)):  # type: ignore[attr-defined]
        epoch_loss = 0.0
        model.train()
        for step, batch in enumerate(train_loader):
            inp = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            ans_ids = batch["answer_ids"]

            with torch.cuda.amp.autocast(enabled=use_amp):
                loss, entropy_val = autoregressive_ce_loss(model, inp, attn, ans_ids, pad_id)
            loss_scaled = loss / grad_accum
            scaler.scale(loss_scaled).backward()
            epoch_loss += loss.item() * len(ans_ids)

            if (step + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)  # type: ignore[attr-defined]

                if blade_ctl:
                    blade_ctl.update(loss.item(), entropy_val, scheduler.get_last_lr()[0])
                if sketch_ctl:
                    sketch_ctl.update(loss.item(), entropy_val)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

                if wb and global_step % cfg.training.logging_steps == 0:  # type: ignore[attr-defined]
                    wb.log(
                        {
                            "train/loss": loss.item(),
                            "train/entropy": entropy_val,
                            "lr": scheduler.get_last_lr()[0],
                            "epoch": epoch,
                        },
                        step=global_step,
                    )

            if cfg.mode == "trial" and global_step >= 10:
                break

        # ---------------- Validation at epoch end -----------------------------
        val_loss, val_acc, preds, golds = evaluate(
            model,
            val_loader,
            tokenizer,
            device,
            pad_id,
            max_batches=2 if cfg.mode == "trial" else -1,
        )
        if wb:
            wb.log(
                {
                    "eval/loss": val_loss,
                    "eval/exact_match": val_acc,
                    "epoch": epoch,
                },
                step=global_step,
            )
            wb.summary["predictions"] = [
                {"gold": g, "pred": p} for g, p in zip(golds, preds)
            ]
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
        print(f"Epoch {epoch}: val_loss={val_loss:.4f}  val_EM={val_acc:.2f}%")
        if cfg.mode == "trial":
            break

    # ---------------- Final evaluation (proxy test) ---------------------------
    _, test_acc, _, _ = evaluate(
        model,
        val_loader,
        tokenizer,
        device,
        pad_id,
        max_batches=2 if cfg.mode == "trial" else -1,
    )

    if wb:
        wb.summary["best_val_exact_match"] = best_val_acc
        wb.summary["best_epoch"] = best_epoch
        wb.summary["test_exact_match"] = test_acc
        wb.summary["optuna_best_params"] = best_params
        print(f"WandB run URL: {wb.url}")
        wb.finish()

    # ---------------- Save artefacts -----------------------------------------
    out_dir = os.path.join(cfg.results_dir, run_id)
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

if __name__ == "__main__":
    main()
