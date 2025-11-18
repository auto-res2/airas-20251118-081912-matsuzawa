"""GSM8K preprocessing pipeline.
This module builds PyTorch DataLoaders *without* ever giving label tokens to the
model during training – labels are kept separate in `answer_ids`."""

import os
from typing import Any, Dict, List, Tuple

import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


###############################################################################
# Helpers                                                                      #
###############################################################################

def build_prompt_and_answer(example: Dict[str, Any]) -> Tuple[str, str]:
    question = example["question"].strip()
    answer_raw = example["answer"].strip()
    gold = answer_raw.split("####")[-1].strip() if "####" in answer_raw else answer_raw
    prompt = f"Question: {question}\n\nAnswer:"
    return prompt, gold


def exact_match_metric(pred: str, gold: str) -> bool:
    return pred.strip() == gold.strip()


###############################################################################
# Dataset wrappers                                                             #
###############################################################################

class GSM8KDataset(torch.utils.data.Dataset):
    def __init__(self, hf_split: Dataset, tokenizer: AutoTokenizer, max_len: int = 1024):
        self.samples: List[Dict[str, Any]] = []
        for ex in hf_split:
            prompt, gold = build_prompt_and_answer(ex)
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)[:max_len]
            gold_ids = tokenizer.encode(" " + gold, add_special_tokens=False)
            self.samples.append(
                {
                    "input_ids": torch.tensor(prompt_ids, dtype=torch.long),
                    "attention_mask": torch.ones(len(prompt_ids), dtype=torch.long),
                    "answer_ids": torch.tensor(gold_ids, dtype=torch.long),
                    "answer_text": gold,
                }
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class Collator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch: List[Dict[str, Any]]):
        max_len = max(x["input_ids"].size(0) for x in batch)
        input_ids, attn = [], []
        ans_ids, ans_txt = [], []
        for item in batch:
            pad_len = max_len - item["input_ids"].size(0)
            input_ids.append(
                torch.cat(
                    [
                        item["input_ids"],
                        torch.full((pad_len,), self.pad_id, dtype=torch.long),
                    ]
                )
            )
            attn.append(
                torch.cat(
                    [
                        item["attention_mask"],
                        torch.zeros(pad_len, dtype=torch.long),
                    ]
                )
            )
            ans_ids.append(item["answer_ids"])
            ans_txt.append(item["answer_text"])
        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attn),
            "answer_ids": ans_ids,
            "answer_text": ans_txt,
        }


###############################################################################
# Public API                                                                   #
###############################################################################

def build_dataloaders(cfg, token_cache_dir: str = ".cache/"):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, cache_dir=token_cache_dir)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    hf_train = load_dataset(
        "openai/gsm8k",
        cfg.dataset.hf_subset,
        split=cfg.dataset.train_split,
        cache_dir=token_cache_dir,
    )
    hf_val = load_dataset(
        "openai/gsm8k",
        cfg.dataset.hf_subset,
        split=cfg.dataset.validation_split,
        cache_dir=token_cache_dir,
    )

    ds_train = GSM8KDataset(hf_train, tokenizer, cfg.dataset.max_seq_length)
    ds_val = GSM8KDataset(hf_val, tokenizer, cfg.dataset.max_seq_length)

    collator = Collator(tokenizer.pad_token_id)
    train_loader = DataLoader(
        ds_train,
        batch_size=cfg.training.per_device_batch_size,
        shuffle=True,
        num_workers=cfg.training.dataloader_num_workers,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=cfg.training.per_device_batch_size,
        shuffle=False,
        num_workers=cfg.training.dataloader_num_workers,
        collate_fn=collator,
    )
    return train_loader, val_loader, tokenizer
