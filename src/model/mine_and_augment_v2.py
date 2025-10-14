import os, argparse, subprocess, pandas as pd, json
import sys

def run_discriminator(backbone, iteration, train_base, train_aug):
    print(f"\n🚀 Iteration {iteration}: Training discriminator...")
    output_dir = f"outputs/discriminator_iter{iteration:02d}"
    cmd = [
        sys.executable,  # ← במקום "python"
        "run_discriminator.py",
        "--backbone", backbone,
        "--train_base_file", train_base,
        "--train_aug_file", train_aug,
        "--output_name", output_dir,
    ]
    subprocess.run(cmd, check=True)

    fn_file = os.path.join("..", "..", "data", "processed", "hard_spam_false_negatives.csv")
    borderline_file = os.path.join("..", "..", "data", "processed", "hard_spam_borderline.csv")
    bottompct_file = os.path.join("..", "..", "data", "processed", "hard_spam_bottompct.csv")

    dfs = []
    counts = {"false_negatives": 0, "borderline": 0, "bottom_p": 0}
    for name, f in zip(counts.keys(), [fn_file, borderline_file, bottompct_file]):
        if os.path.exists(f):
            df = pd.read_csv(f)
            counts[name] = len(df)
            dfs.append(df)
    if dfs:
        df_all = pd.concat(dfs).drop_duplicates(subset=["message"])
        hard_file = os.path.join("..", "..", "data", "hard_examples.csv")
        df_all.to_csv(hard_file, index=False)
        print(f"✅ Saved hard examples for iteration {iteration} → {hard_file} ({len(df_all)} samples)")
    else:
        print("⚠️ No hard examples found.")
    return counts

def run_generator(iteration, num_to_generate, mode="hard"):
    print(f"\n✍️ Iteration {iteration}: Generating augmented spam ({mode})...")
    out_file = os.path.join("..", "..", "data", f"augmented_spam_iter{iteration:02d}.csv")
    cmd = ["python", "llm_generator_v2.py", "--mode", mode, "--num_messages", str(num_to_generate), "--output", out_file]
    subprocess.run(cmd, check=True)
    print(f"✅ Augmented spam saved to {out_file}")
    return out_file

def append_summary(iteration, metrics, counts, tau, new_spam_generated):
    summary_file = os.path.join("..", "..", "data", "mine_and_augment_summary.json")
    summary = []
    if os.path.exists(summary_file):
        with open(summary_file, "r") as f:
            try: summary = json.load(f)
            except Exception: summary = []
    summary.append({
        "iteration": iteration,
        "accuracy": metrics.get("accuracy"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "f1": metrics.get("f1"),
        "tau_prob": tau.get("tau_prob") if isinstance(tau, dict) else None,
        "tau_energy": tau.get("tau_energy") if isinstance(tau, dict) else None,
        "false_negatives": counts["false_negatives"],
        "borderline": counts["borderline"],
        "bottom_p": counts["bottom_p"],
        "new_spam_generated": new_spam_generated
    })
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"📊 Summary updated → {summary_file}")
    return summary

def should_stop_early(summary, patience=2):
    # Stop if F1 didn't improve for `patience` consecutive iterations
    if len(summary) <= patience: return False
    best_so_far = -1.0
    drops = 0
    for i in range(len(summary)):
        f1 = summary[i].get("f1") or 0.0
        if f1 > best_so_far + 1e-6:
            best_so_far = f1
            drops = 0
        else:
            drops += 1
        if drops >= patience:
            return True
    return False

def merge_augmented_for_next_iter(base_aug_path, new_aug_path):
    df_base = pd.read_csv(base_aug_path) if os.path.exists(base_aug_path) else pd.DataFrame(columns=["message","label"])
    df_new  = pd.read_csv(new_aug_path)
    df_new["label"] = 1  # בטוח ספאם
    df_merged = pd.concat([df_base, df_new], ignore_index=True).drop_duplicates(subset=["message"])
    df_out = os.path.join("..", "..", "data", "processed", "train_sms_mistral_augmented.csv")
    df_merged.to_csv(df_out, index=False)
    print(f"🔁 Merged augmented data → {df_out} (total {len(df_merged)} rows)")
    return df_out

# הוסף את זה בתוך mine_and_augment_v2.py
def run_pipeline(iters=5,
                 per_iter_new=200,
                 patience=2,
                 backbone="../../models/deberta_v3-base",
                 train_base="../../data/processed/train_spam_merged_dedup.csv",
                 train_aug="../../data/processed/train_spam_merged_dedup.csv"):
    current_aug = train_aug
    for it in range(1, iters + 1):
        counts = run_discriminator(backbone, it, train_base, current_aug)
        gen_file = run_generator(it, per_iter_new, mode="hard")

        # קרא את המטריקות והספים מהריצה
        run_dir = os.path.join("../../models", f"outputs/discriminator_iter{it:02d}")
        metrics_file = os.path.join(run_dir, "metrics.json")
        tau_file = os.path.join(run_dir, "threshold.json")
        metrics = {}; tau = {}
        if os.path.exists(metrics_file):
            with open(metrics_file, "r", encoding="utf-8") as f:
                try: metrics = json.load(f)
                except: metrics = {}
        if os.path.exists(tau_file):
            with open(tau_file, "r", encoding="utf-8") as f:
                try: tau = json.load(f)
                except: tau = {}

        summary = append_summary(it, metrics, counts, tau, per_iter_new)

        # עצירה אוטומטית אם אין שיפור
        if should_stop_early(summary, patience=patience):
            print(f"⛳ Early stopping at iteration {it} (no F1 improvement for {patience} iters).")
            break

        # מיזוג האוגמנטציות לאיטרציה הבאה
        current_aug = merge_augmented_for_next_iter(current_aug, gen_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--per_iter_new", type=int, default=500)
    parser.add_argument("--backbone", type=str, default="../../models/deberta_v3-base")
    parser.add_argument("--train_base", type=str, default="../../data/processed/train_spam_merged_dedup.csv")
    parser.add_argument("--train_aug", type=str, default="../../data/processed/train_spam_merged_dedup.csv")
    parser.add_argument("--patience", type=int, default=2)
    args = parser.parse_args()

    run_pipeline(
        iters=args.iters,
        per_iter_new=args.per_iter_new,
        patience=args.patience,
        backbone=args.backbone,
        train_base=args.train_base,
        train_aug=args.train_aug,
    )

if __name__ == "__main__":
    main()
