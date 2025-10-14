# --- replace the imports at top ---
import os, json, shutil
from typing import Tuple

import torch
from transformers import (
    AutoModelForSequenceClassification,
    DebertaV2Tokenizer,   # <-- only slow tokenizer
)

HUB_MODEL_ID = "microsoft/deberta-v3-base"

# Keep only the files needed for slow (SentencePiece) tokenizer
REQUIRED_TOKENIZER_FILES = [
    "tokenizer_config.json",
    "special_tokens_map.json",
    "spm.model",               # critical for slow
]

SPECIAL_TOKENS_MAP_CONTENT = {
    "bos_token": "[CLS]",
    "cls_token": "[CLS]",
    "eos_token": "[SEP]",
    "mask_token": "[MASK]",
    "pad_token": "[PAD]",
    "sep_token": "[SEP]",
    "unk_token": "[UNK]"
}

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _file_exists(dirpath: str, fname: str) -> bool:
    return os.path.isfile(os.path.join(dirpath, fname))

def _need_download(local_dir: str) -> bool:
    if not os.path.isdir(local_dir):
        return True
    for f in REQUIRED_TOKENIZER_FILES:
        if not _file_exists(local_dir, f):
            return True
    has_pt = any(n.endswith(".bin") or n.endswith(".safetensors") for n in os.listdir(local_dir))
    return not has_pt

def _write_special_tokens_map(local_dir: str):
    path = os.path.join(local_dir, "special_tokens_map.json")
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(SPECIAL_TOKENS_MAP_CONTENT, f, indent=2, ensure_ascii=False)

def _download_and_cache_to_local(local_dir: str):
    """Download ONLY slow tokenizer + model to avoid fast conversion."""
    _ensure_dir(local_dir)
    # slow tokenizer (SentencePiece)
    tok = DebertaV2Tokenizer.from_pretrained(HUB_MODEL_ID)
    tok.save_pretrained(local_dir)
    _write_special_tokens_map(local_dir)
    # model weights
    model_tmp = AutoModelForSequenceClassification.from_pretrained(HUB_MODEL_ID, num_labels=2)
    model_tmp.save_pretrained(local_dir)

def _load_tokenizer(local_dir: str) -> Tuple[object, str]:
    """Always load slow tokenizer to avoid conversion and protobuf/tiktoken deps."""
    tok = DebertaV2Tokenizer.from_pretrained(local_dir)
    return tok, "slow"


def _is_local_path(p: str) -> bool:
    p = os.path.expanduser(p)
    return (os.path.sep in p) or p.startswith((".", "~", "/", "\\"))

def load_model_and_tokenizer(backbone_path_or_id: str, num_labels: int = 2):
    # נרמל נתיב אם נראה כמו נתיב
    if _is_local_path(backbone_path_or_id):
        local_dir = os.path.abspath(os.path.expanduser(backbone_path_or_id))
        os.makedirs(local_dir, exist_ok=True)
        if _need_download(local_dir):
            print(f"[deberta_bootstrap] Downloading to local dir: {local_dir}")
            _download_and_cache_to_local(local_dir)
        tokenizer = DebertaV2Tokenizer.from_pretrained(local_dir)
        model = AutoModelForSequenceClassification.from_pretrained(local_dir, num_labels=num_labels)
        return tokenizer, model

    # אחרת – זה Hub id
    print(f"[deberta_bootstrap] Loading from Hub: {backbone_path_or_id}")
    tokenizer = DebertaV2Tokenizer.from_pretrained(backbone_path_or_id)
    model = AutoModelForSequenceClassification.from_pretrained(backbone_path_or_id, num_labels=num_labels)
    return tokenizer, model
