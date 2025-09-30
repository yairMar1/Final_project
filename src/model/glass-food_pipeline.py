"""
GLASS-FOOD reproduction patch
--------------------------------
This repo-friendly bundle gives you three entry points that match the paper's pipeline:

1) augment_glassfood.py  – cosine-sim augmentation (one targeted token) for **all** messages (ham+spam).
2) train_glassfood.py    – train RoBERTa discriminator **without class-weights**; pick a message-level
                           pseudo-labeling threshold (τ) on the validation set (max-F1).
3) eval_glassfood.py     – evaluate on test set using τ and report metrics (+ optional BLEU & RIS hooks).

Assumes your existing structure (relative paths) and reuses your TextAugmenter from roberta_aug.py
(with small safety fixes). If you prefer a single script, keep these as modules or run as CLIs.

Paths expect ../../data/processed/ and ../../models/ etc. Adjust to your tree if needed.
"""

# ==========================
# augment_glassfood.py
# ==========================

import os
import json
import math
import pandas as pd
from typing import Optional

import torch
from transformers import AutoTokenizer

# Reuse your augmenter implementation
# Make sure roberta_aug.py is importable (same folder) or add to PYTHONPATH
from roberta_aug import TextAugmenter  # your file

PROCESSED_DATA_DIR = "../../data/processed/"
INPUT_FILE = "train_sms.csv"          # original (ham/spam) SMS Spam Collection
OUTPUT_FILE = "train_sms_glassfood_aug_roberta.csv"  # will contain original + 1 augmented per message

TARGET_ONE_AUG_PER_MSG = True  # paper enriches dataset via *paired* generated texts
MAX_LEN = 512


def augment_all_messages(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    df = df.copy()
    assert {"message", "label"}.issubset(df.columns)

    augmenter = TextAugmenter()

    texts = df["message"].astype(str).fillna("").tolist()

    # Batch augmentation using your replace_farthest_token_batch
    # One augmented sample per original sample (ham + spam), 1:1 pairing
    batch_size = 64
    augmented = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        out = augmenter.replace_farthest_token_batch(batch)
        augmented.extend(out)

    df_aug = pd.DataFrame({
        "message": augmented,
        "label": df["label"].tolist()
    })

    # Keep only messages that actually changed (fallback to original if none changed)
    changed = df_aug["message"] != df["message"].values
    if changed.sum() == 0:
        # If nothing changed (shouldn't happen), fall back to original
        df_out = pd.concat([df, df], ignore_index=True)
    else:
        # Include original + changed augmented
        df_aug = df_aug[changed]
        df_out = pd.concat([df, df_aug], ignore_index=True)

    # Shuffle
    df_out = df_out.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return df_out


def main_augment():
    in_path = os.path.join(PROCESSED_DATA_DIR, INPUT_FILE)
    out_path = os.path.join(PROCESSED_DATA_DIR, OUTPUT_FILE)

    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Missing input file: {in_path}")

    df = pd.read_csv(in_path)
    print("Loaded:", df.shape, df["label"].value_counts().to_dict())
    df_out = augment_all_messages(df)
    print("Augmented set:", df_out.shape, df_out["label"].value_counts().to_dict())

    df_out.to_csv(out_path, index=False)
    print("Saved to:", out_path)


# ==========================
# train_glassfood.py
# ==========================

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    IntervalStrategy,
)

import torch
from torch import nn

AUG_FILE = OUTPUT_FILE  # from the augmentation step
TEST_FILE = "test_sms.csv"
MODEL_DIR = "../../models/glassfood_discriminator/"
LOG_DIR = "../../logs/glassfood_discriminator/"
ROBERTA = "roberta-base"
MAX_LENGTH = 512
EPOCHS = 5
SEED = 42
BSZ_TRAIN = 16
BSZ_EVAL = 64


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["message"] = df["message"].fillna("").astype(str)
    df["label"] = df["label"].astype(int)
    return df


class SMSDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)


def compute_metrics(pred):
    labels = pred.label_ids
    logits = pred.predictions
    preds = logits.argmax(axis=1)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', pos_label=1, zero_division=0)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": p, "recall": r}


class PlainTrainer(Trainer):
    """RoBERTa discriminator with **no class-weights** (per paper)."""
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def fit_and_pick_threshold(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    tokenizer = AutoTokenizer.from_pretrained(ROBERTA)
    model = AutoModelForSequenceClassification.from_pretrained(ROBERTA, num_labels=2)

    def _enc(df):
        return tokenizer(df["message"].tolist(), truncation=True, padding=True, max_length=MAX_LENGTH)

    ds_train = SMSDataset(_enc(train_df), train_df["label"].tolist())
    ds_val   = SMSDataset(_enc(val_df),   val_df["label"].tolist())
    ds_test  = SMSDataset(_enc(test_df),  test_df["label"].tolist())

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    args = TrainingArguments(
        output_dir=MODEL_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BSZ_TRAIN,
        per_device_eval_batch_size=BSZ_EVAL,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_dir=LOG_DIR,
        logging_steps=100,
        eval_strategy=IntervalStrategy.EPOCH,
        save_strategy=IntervalStrategy.EPOCH,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
        seed=SEED,
    )

    trainer = PlainTrainer(
        model=model,
        args=args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()

    # --- Message-level pseudo-label threshold τ (maximize F1 on validation) ---
    val_logits = trainer.predict(ds_val).predictions
    val_probs_spam = torch.softmax(torch.tensor(val_logits), dim=1)[:, 1].numpy()
    best_f1, best_tau = -1.0, 0.5
    y_true = val_df["label"].values
    for tau in np.linspace(0.05, 0.95, 19):
        y_pred = (val_probs_spam >= tau).astype(int)
        f1 = precision_recall_fscore_support(y_true, y_pred, average='binary', pos_label=1, zero_division=0)[2]
        if f1 > best_f1:
            best_f1, best_tau = f1, float(tau)

    # Persist model + threshold
    trainer.save_model(os.path.join(MODEL_DIR, "best_model"))
    tokenizer.save_pretrained(os.path.join(MODEL_DIR, "best_model"))
    with open(os.path.join(MODEL_DIR, "threshold.json"), "w") as f:
        json.dump({"tau": best_tau}, f)

    print(f"Selected τ={best_tau:.2f} (max F1 on validation)")

    # quick test preview
    test_logits = trainer.predict(ds_test).predictions
    test_probs_spam = torch.softmax(torch.tensor(test_logits), dim=1)[:, 1].numpy()
    y_hat = (test_probs_spam >= best_tau).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(test_df["label"].values, y_hat, average='binary', pos_label=1, zero_division=0)
    acc = accuracy_score(test_df["label"].values, y_hat)
    print({"accuracy": acc, "f1": f1, "precision": p, "recall": r})


def main_train():
    train_path = os.path.join(PROCESSED_DATA_DIR, AUG_FILE)
    test_path  = os.path.join(PROCESSED_DATA_DIR, TEST_FILE)
    df_train_all = load_csv(train_path)
    df_test = load_csv(test_path)

    # stratified split into train/val
    train_df, val_df = train_test_split(
        df_train_all, test_size=0.1, random_state=SEED, stratify=df_train_all["label"]
    )

    print("Train/Val/Test:", len(train_df), len(val_df), len(df_test))
    fit_and_pick_threshold(train_df, val_df, df_test)


# ==========================
# eval_glassfood.py
# ==========================

from sklearn.metrics import classification_report


def main_eval():
    # Load test
    test_path  = os.path.join(PROCESSED_DATA_DIR, TEST_FILE)
    df_test = load_csv(test_path)

    # Load model + τ
    tok = AutoTokenizer.from_pretrained(os.path.join(MODEL_DIR, "best_model"))
    mdl = AutoModelForSequenceClassification.from_pretrained(os.path.join(MODEL_DIR, "best_model"))
    with open(os.path.join(MODEL_DIR, "threshold.json")) as f:
        tau = json.load(f)["tau"]

    enc = tok(df_test["message"].tolist(), truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt")
    with torch.no_grad():
        logits = mdl(**{k: v for k, v in enc.items()}).logits
    probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    y_hat = (probs >= tau).astype(int)

    print(classification_report(df_test["label"].values, y_hat, digits=4))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GLASS-FOOD pipeline runner")
    parser.add_argument("stage", choices=["augment", "train", "eval"], help="which stage to run")
    args = parser.parse_args()

    if args.stage == "augment":
        main_augment()
    elif args.stage == "train":
        main_train()
    else:
        main_eval()
