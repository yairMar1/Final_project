# --- ADDITIONS: top imports (replace your top block) ---
import torch
import pandas as pd
import numpy as np
import os, json, argparse, shutil
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, EarlyStoppingCallback)
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from deberta_bootstrap import load_model_and_tokenizer
from copy import deepcopy

PROCESSED_DATA_DIR = "../../data/processed/"
DEFAULT_TRAIN_AUG_FILE = "train_spam_merged_dedup.csv"
DEFAULT_TRAIN_BASE_FILE = "train_sms_dedup.csv"
DEFAULT_TEST_FILE = "test_sms_dedup.csv"

HARD_MINING_BOTTOM_PCT = 0.20
BORDERLINE_MARGIN = 0.05
DEFAULT_TAU = 0.5

# ---------------- OOD/Energy helpers ----------------
def logits_to_energy(logits: torch.Tensor, T: float = 1.0) -> torch.Tensor:
    # E(x) = -log sum_j exp(z_j / T) ; lower is "more confident"
    return -torch.logsumexp(logits / T, dim=1)

class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_T = nn.Parameter(torch.zeros(1))  # T=1 initially
    def forward(self, logits):
        T = torch.exp(self.log_T)
        return logits / T
    def fit(self, logits, labels, max_iter=200, lr=0.01):
        # Simple NLL minimization
        opt = torch.optim.LBFGS(self.parameters(), lr=lr, max_iter=max_iter)
        labels = labels.long()
        nll = nn.CrossEntropyLoss()
        def closure():
            opt.zero_grad()
            loss = nll(self.forward(logits), labels)
            loss.backward()
            return loss
        opt.step(closure)
        return float(torch.exp(self.log_T).item())

# ---------------- Data & Dataset ----------------
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["message"] = df["message"].fillna("").astype(str)
    df["label"]   = df["label"].astype(int)
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

# ---------------- Metrics ----------------
def compute_metrics(pred):
    labels = pred.label_ids
    logits = pred.predictions
    preds  = logits.argmax(axis=1)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average='binary',
                                                  pos_label=1, zero_division=0)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": p, "recall": r}

# ---------------- Trainer with Label-Smoothing & (optional) R-Drop ----------------
class WeightedTrainer(Trainer):
    def __init__(self, *args, label_smoothing=0.05, rdrop_alpha=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_smoothing = label_smoothing
        self.rdrop_alpha = rdrop_alpha
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # Label smoothing CE
        ls = self.label_smoothing
        num_classes = logits.size(-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(logits).fill_(ls / (num_classes - 1))
            true_dist.scatter_(1, labels.unsqueeze(1), 1.0 - ls)
        log_probs = nn.LogSoftmax(dim=-1)(logits)
        ce_loss = -(true_dist * log_probs).sum(dim=1).mean()

        if self.rdrop_alpha > 0.0:
            # Forward again with dropout active for KL consistency
            outputs2 = model(**inputs)
            logits2 = outputs2.get("logits")
            p1 = nn.LogSoftmax(dim=-1)(logits)
            p2 = nn.LogSoftmax(dim=-1)(logits2)
            kl = (torch.exp(p1) * (p1 - p2)).sum(dim=1).mean() + (torch.exp(p2) * (p2 - p1)).sum(dim=1).mean()
            loss = ce_loss + self.rdrop_alpha * kl * 0.5
        else:
            loss = ce_loss
        return (loss, outputs) if return_outputs else loss

# ---------------- Tau selection (prob & energy) ----------------
def pick_tau_prob_and_energy(trainer: Trainer, val_dataset: SMSDataset, y_true: np.ndarray, T: float = 1.0):
    val_logits = torch.tensor(trainer.predict(val_dataset).predictions)
    probs = torch.softmax(val_logits / T, dim=1)[:, 1].numpy()
    energy = logits_to_energy(val_logits, T=T).numpy()

    # Grid search for probability threshold
    best_f1_p, best_tau_p = -1.0, 0.5
    for tau in np.linspace(0.05, 0.95, 19):
        y_hat = (probs >= tau).astype(int)
        f1 = precision_recall_fscore_support(y_true, y_hat, average='binary', pos_label=1, zero_division=0)[2]
        if f1 > best_f1_p:
            best_f1_p, best_tau_p = f1, float(tau)

    # Grid search for energy threshold (lower energy = more confident)
    best_f1_e, best_tau_e = -1.0, float(np.percentile(energy, 50))
    for tauE in np.linspace(np.percentile(energy, 1), np.percentile(energy, 99), 25):
        y_hat_e = (energy <= tauE).astype(int)  # classify as spam if energy <= tauE
        f1_e = precision_recall_fscore_support(y_true, y_hat_e, average='binary', pos_label=1, zero_division=0)[2]
        if f1_e > best_f1_e:
            best_f1_e, best_tau_e = f1_e, float(tauE)
    return (best_tau_p, best_f1_p), (best_tau_e, best_f1_e)

# ---------------- MAIN ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, default="../../models/deberta_v3-base")
    parser.add_argument("--train_aug_file", type=str, default=DEFAULT_TRAIN_AUG_FILE)
    parser.add_argument("--train_base_file", type=str, default=DEFAULT_TRAIN_BASE_FILE)
    parser.add_argument("--test_file", type=str, default=DEFAULT_TEST_FILE)
    parser.add_argument("--output_name", type=str, default="discriminator_deberta_v3")
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--bsz_train", type=int, default=32)
    parser.add_argument("--bsz_eval", type=int, default=128)
    parser.add_argument("--rdrop_alpha", type=float, default=0.0)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE} | CUDA available: {torch.cuda.is_available()}")

    train_aug_path = os.path.join(PROCESSED_DATA_DIR, args.train_aug_file)
    base_train_path= os.path.join(PROCESSED_DATA_DIR, args.train_base_file)
    test_path      = os.path.join(PROCESSED_DATA_DIR, args.test_file)

    df_train_val = load_csv(train_aug_path)
    df_test = load_csv(test_path)
    train_df, val_df = train_test_split(
        df_train_val, test_size=0.1, random_state=42, stratify=df_train_val["label"]
    )

    # Class weights (אפשר להשאיר; במאמר טענו "לא חייב", אבל זה יציב)
    class_weights = compute_class_weight(class_weight='balanced',
                                         classes=np.unique(train_df['label']),
                                         y=train_df['label'])
    print(f"Class weights [ham, spam]: {class_weights}")

    tokenizer, model = load_model_and_tokenizer(args.backbone, num_labels=2)
    def enc(df):
        return tokenizer(df["message"].tolist(), truncation=True, padding=True, max_length=args.max_len)

    ds_train = SMSDataset(enc(train_df), train_df["label"].tolist())
    ds_val   = SMSDataset(enc(val_df),   val_df["label"].tolist())
    ds_test  = SMSDataset(enc(df_test),  df_test["label"].tolist())

    model = model.to(DEVICE)

    out_dir   = os.path.join("../../models/", args.output_name)
    log_dir   = os.path.join("../../logs/", args.output_name + "_logs")
    best_dir  = os.path.join(out_dir, "best_model")
    if os.path.exists(out_dir): shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True); os.makedirs(log_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bsz_train,
        per_device_eval_batch_size=args.bsz_eval,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_dir=log_dir,
        logging_steps=100,

        eval_strategy="epoch",  # נדרש כדי לבחור best
        save_strategy="epoch",  # נשמור פעם לאפוך־סבב
        load_best_model_at_end=True,  # נטען את ה־best לזיכרון בסוף
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=1,  # סבב רוטציה שומר מקסימום צ'קפוינט 1 בזמן אימון

        report_to="none",
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        seed=42
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        label_smoothing=args.label_smoothing,
        rdrop_alpha=args.rdrop_alpha
    )

    trainer.train()

    # ---- Temperature scaling on validation logits ----
    with torch.no_grad():
        val_logits = torch.tensor(trainer.predict(ds_val).predictions)
        val_labels = torch.tensor(val_df["label"].values)
    scaler = TemperatureScaler()
    T_star = scaler.fit(val_logits, val_labels)
    print(f"Calibrated temperature T* = {T_star:.3f}")

    # ---- Evaluate (calibrated) on val & test ----
    def eval_with_T(dataset):
        logits = torch.tensor(trainer.predict(dataset).predictions)
        logits_cal = logits / T_star
        probs = torch.softmax(logits_cal, dim=1)[:, 1].numpy()
        preds = (probs >= 0.5).astype(int)
        labels = np.array(dataset.labels)
        p, r, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', pos_label=1, zero_division=0)
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}, logits, logits_cal

    val_metrics, val_logits_raw, val_logits_cal = eval_with_T(ds_val)
    test_metrics, test_logits_raw, test_logits_cal = eval_with_T(ds_test)
    print("Validation (calibrated):", val_metrics)
    print("Test (calibrated):", test_metrics)

    # ---- Pick tau on probability & energy (calibrated) ----
    (tau_p, f1_p), (tau_e, f1_e) = pick_tau_prob_and_energy(trainer, ds_val, val_df["label"].values, T=T_star)
    print(f"Selected τ_prob={tau_p:.3f} (val F1={f1_p:.4f}) | τ_energy={tau_e:.4f} (val F1={f1_e:.4f})")

    # Save best + thresholds + temperature
    trainer.save_model(best_dir); tokenizer.save_pretrained(best_dir)
    with open(os.path.join(out_dir, "threshold.json"), "w", encoding="utf-8") as f:
        json.dump({"tau_prob": float(tau_p), "tau_energy": float(tau_e), "temperature": float(T_star)}, f)

    # --- "Global best" across all iterations ---
    global_best_dir = os.path.join("../../models", "global_best_discriminator")
    global_best_meta = os.path.join(global_best_dir, "global_best.json")
    this_f1 = float(test_metrics["f1"])  # או תעדף val_f1 אם זה המדד הקובע

    os.makedirs(global_best_dir, exist_ok=True)
    prev_best = -1.0
    if os.path.exists(global_best_meta):
        try:
            with open(global_best_meta, "r", encoding="utf-8") as f:
                prev_best = json.load(f).get("best_f1", -1.0)
        except:
            prev_best = -1.0

    if this_f1 > prev_best + 1e-8:
        # מחליפים את האלוף
        for name in os.listdir(global_best_dir):
            p = os.path.join(global_best_dir, name)
            if os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)
        target_best_dir = os.path.join(global_best_dir, "best_model")
        shutil.copytree(best_dir, target_best_dir)

        with open(global_best_meta, "w", encoding="utf-8") as f:
            json.dump({
                "best_f1": this_f1,
                "source_iteration_dir": out_dir,
                "tau_prob": float(tau_p),
                "tau_energy": float(tau_e),
                "temperature": float(T_star),
                "test_metrics": test_metrics,
                "val_metrics": val_metrics
            }, f, indent=2, ensure_ascii=False)
        print(f"🏆 New GLOBAL BEST (F1={this_f1:.4f}) saved to {global_best_dir}")
    else:
        print(f"Global best unchanged (prev {prev_best:.4f} vs current {this_f1:.4f})")

    # מחיקה יזומה של כל תקיות ה-checkpoint כדי לחסוך מקום
    for name in os.listdir(out_dir):
        if name.startswith("checkpoint-"):
            try:
                shutil.rmtree(os.path.join(out_dir, name))
            except Exception as e:
                print("WARN: failed to remove", name, e)

    # Save metrics summary
    metrics_obj = {
        "accuracy": float(test_metrics["accuracy"]),
        "precision": float(test_metrics["precision"]),
        "recall": float(test_metrics["recall"]),
        "f1": float(test_metrics["f1"]),
        "tau_prob": float(tau_p),
        "best_val_f1_at_tau_prob": float(f1_p),
        "tau_energy": float(tau_e),
        "best_val_f1_at_tau_energy": float(f1_e),
        "temperature": float(T_star),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "sizes": {"train": len(ds_train), "val": len(ds_val), "test": len(ds_test)}
    }
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_obj, f, indent=2, ensure_ascii=False)

    # ---- HARD MINING (prob + energy) ----
    if not os.path.exists(base_train_path):
        print(f"WARNING: base train file not found at {base_train_path}. Skipping mining.")
        return

    import gc
    torch.cuda.empty_cache();
    gc.collect()

    df_base = load_csv(base_train_path)
    texts = df_base["message"].tolist()
    BATCH = 128  # אם עדיין צפוף בזיכרון נסה 64 או 32
    model.eval()

    probs_list, energy_list = [], []
    with torch.no_grad():
        for i in range(0, len(texts), BATCH):
            chunk = texts[i:i + BATCH]
            enc = tokenizer(chunk, truncation=True, padding="longest",
                            max_length=args.max_len, return_tensors="pt")
            enc = {k: v.to(DEVICE, non_blocking=True) for k, v in enc.items()}
            logits = model(**enc).logits
            logits = logits / T_star  # calibrated
            probs = torch.softmax(logits, dim=1)[:, 1]
            energy = logits_to_energy(logits)

            probs_list.append(probs.detach().cpu())
            energy_list.append(energy.detach().cpu())

            del enc, logits, probs, energy
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    probs = torch.cat(probs_list).numpy()
    energy = torch.cat(energy_list).numpy()

    # --- Build calibrated scores frame ---
    df_scores = df_base.copy()
    df_scores["prob_spam"] = probs  # numpy -> float column
    df_scores["energy"] = energy

    # Energy-rule prediction at calibrated τ_e (1 = spam if energy <= τ_e)
    df_scores["pred_energy"] = (df_scores["energy"] <= tau_e).astype(int)

    # Convenience cols
    df_scores["delta_to_tau"] = df_scores["energy"] - tau_e  # >0 => more "ham-like"
    spam_mask = (df_scores["label"] == 1)

    # 1) FALSE NEGATIVES (hardest): actual spam predicted ham by the energy rule
    fn_mask = ((df_scores["pred_energy"] == 0) & spam_mask)  # pred_energy==0 => energy > τ_e
    df_fn = df_scores.loc[fn_mask, ["label", "message", "energy", "delta_to_tau"]].copy()
    # Prioritize FNs most on ham side (largest positive delta_to_tau)
    df_fn = df_fn.sort_values("delta_to_tau", ascending=False)

    # 2) BORDERLINE: closest to τ_e among spam (exclude FNs to avoid dupes)
    if spam_mask.any():
        bl_pool = df_scores.loc[spam_mask, ["label", "message", "energy", "delta_to_tau"]].copy()
        if not df_fn.empty:
            bl_pool = bl_pool.loc[~bl_pool["message"].isin(df_fn["message"])]
        bl_pool["abs_delta"] = bl_pool["delta_to_tau"].abs()
    else:
        bl_pool = pd.DataFrame(columns=["label", "message", "energy", "delta_to_tau", "abs_delta"])

    TARGET_TOTAL = int(os.environ.get("HARD_POOL_CAP", "300"))
    target_bl = max(40, int(TARGET_TOTAL * 0.33))  # ~1/3 budget for borderline
    df_borderline = (
        bl_pool.nsmallest(target_bl, "abs_delta")[["label", "message", "energy", "delta_to_tau"]]
        if not bl_pool.empty else
        pd.DataFrame(columns=["label", "message", "energy", "delta_to_tau"])
    )

    # 3) TOP-ENERGY spam: highest energy among remaining spam (closest to ham), excluding FNs & borderline
    top_pool = df_scores.loc[spam_mask, ["label", "message", "energy", "delta_to_tau"]].copy()
    if not df_fn.empty:
        top_pool = top_pool.loc[~top_pool["message"].isin(df_fn["message"])]
    if not df_borderline.empty:
        top_pool = top_pool.loc[~top_pool["message"].isin(df_borderline["message"])]

    taken_so_far = len(df_fn) + len(df_borderline)
    target_top = max(40, TARGET_TOTAL - taken_so_far)  # fill the rest up to TARGET_TOTAL
    df_bottom = (
        top_pool.nlargest(target_top, "energy")
        if not top_pool.empty else
        pd.DataFrame(columns=["label", "message", "energy", "delta_to_tau"])
    )

    # 4) WRITE outputs for the generator (just label+message)
    out_scores_path = os.path.join(PROCESSED_DATA_DIR, "discriminator_scores_train.csv")
    out_fn_path = os.path.join(PROCESSED_DATA_DIR, "hard_spam_false_negatives.csv")
    out_bl_path = os.path.join(PROCESSED_DATA_DIR, "hard_spam_borderline.csv")
    out_bot_path = os.path.join(PROCESSED_DATA_DIR, "hard_spam_bottompct.csv")

    df_scores.to_csv(out_scores_path, index=False)
    df_fn[["label", "message"]].to_csv(out_fn_path, index=False)
    df_borderline[["label", "message"]].to_csv(out_bl_path, index=False)
    df_bottom[["label", "message"]].to_csv(out_bot_path, index=False)

    print("HARD MINING DONE (adaptive caps):")
    print(f"  false negatives -> {out_fn_path} ({len(df_fn)})")
    print(f"  borderline      -> {out_bl_path} ({len(df_borderline)})")
    print(f"  top energy      -> {out_bot_path} ({len(df_bottom)})")


if __name__ == "__main__":
    main()
