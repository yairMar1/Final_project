import os, sys, json, time, subprocess, argparse, re
import pandas as pd
from pathlib import Path

# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------

def _cwd():
    """Src/model as working dir is assumed; make paths explicit & consistent."""
    return Path(__file__).resolve().parent

def _data_dir():
    return _cwd().parents[2] / "data"   # ../../data from src/model

def _processed_dir():
    return _data_dir() / "processed"

def _ensure_parent(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

# --------------------------------------------------------------------------------------
# Discriminator training + mining
# --------------------------------------------------------------------------------------

VAL_CAL_RE = re.compile(
    r"Validation \(calibrated\):\s*\{[^}]*'f1':\s*([0-9]*\.?[0-9]+)",
    re.IGNORECASE
)

def run_discriminator(backbone: str, iteration: int, train_base: str, train_aug: str):
    print(f"\n🚀 Iteration {iteration}: Training discriminator...")
    output_name = f"outputs/discriminator_iter{iteration:02d}"

    # 1) Run and STREAM logs so you see tqdm/progress
    cmd = [
        sys.executable,
        "run_discriminator.py",
        "--backbone", backbone,
        "--train_base_file", train_base,
        "--train_aug_file",  train_aug,
        "--output_name",     output_name,
    ]
    # No capture_output: you see progress live in the console
    subprocess.run(cmd, check=True, cwd=_cwd())

    # 2) Read metrics.json written by run_discriminator.py
    #    (The script saves under ../../models/<output_name>/metrics.json)
    run_dir = _cwd().parents[1] / "models" / output_name
    metrics_path = run_dir / "metrics.json"

    val_f1 = 0.0
    if metrics_path.exists():
        try:
            m = json.loads(metrics_path.read_text(encoding="utf-8"))
            # use validation F1 if present; fall back to test F1
            val_f1 = float(m.get("val_metrics", {}).get("f1", m.get("f1", 0.0)))
            print(f"✅ Validation F1 (from metrics.json): {val_f1:.4f}")
        except Exception as e:
            print("⚠️ Failed to parse metrics.json:", e)

    # Gather mined files to make hard_examples.csv (unchanged)
    fn_file         = _processed_dir() / "hard_spam_false_negatives.csv"
    borderline_file = _processed_dir() / "hard_spam_borderline.csv"
    bottompct_file  = _processed_dir() / "hard_spam_bottompct.csv"

    counts = {"false_negatives": 0, "borderline": 0, "bottom_p": 0}
    hard_frames = []
    for key, fp in [("false_negatives", fn_file),
                    ("borderline", borderline_file),
                    ("bottom_p", bottompct_file)]:
        if fp.exists():
            df = pd.read_csv(fp)
            counts[key] = len(df)
            hard_frames.append(df)

    if hard_frames:
        df_all = pd.concat(hard_frames, ignore_index=True).drop_duplicates(subset=["message"])
        hard_file = _data_dir() / "hard_examples.csv"
        _ensure_parent(hard_file)
        df_all.to_csv(hard_file, index=False)
        print(f"✅ Saved hard examples for iteration {iteration} → {hard_file} ({len(df_all)} samples)")
    else:
        print("⚠️ No hard examples found.")

    return val_f1, counts

def mine_hard_examples():
    """
    Mining is already performed inside run_discriminator.py.
    This function is kept for pipeline readability and future hooks.
    """
    return

# --------------------------------------------------------------------------------------
# Generator
# --------------------------------------------------------------------------------------

def run_generator(iter_idx: int, num_msgs: int, mode: str = "hard") -> str:
    """
    Calls llm_generator_v2.py (Mistral-7B-Instruct) with speed-friendly hard-mode flags.
    Returns the output CSV path (relative to project root).
    """
    out = _data_dir() / f"augmented_spam_iter{iter_idx:02d}.csv"
    cmd = [
        sys.executable,                 # use same interpreter (venv)
        "llm_generator_v2.py",
        "--mode", mode,
        "--num_messages", str(num_msgs),
        "--output", str(out),
    ]
    if mode == "hard":
        # Speed knobs to keep hard-mode fast
        cmd += [
            "--hard_files", "../../data/hard_examples.csv",  # << use the exact mined set
            "--hard_prompts", "12",
            "--hard_few", "3",
            "--per_prompt", "8",
            "--batch_size", "16",
            "--max_new_tokens", "32",
        ]

    print(f"✍️ Iteration {iter_idx}: Generating augmented spam ({mode})...")
    subprocess.run(cmd, check=True, cwd=_cwd())
    print(f"✅ Generated {num_msgs} messages in {mode} mode → saved to {out}")
    return str(out)

# --------------------------------------------------------------------------------------
# Merge augmented → next train file
# --------------------------------------------------------------------------------------

def merge_augmented_into_train(new_aug_path: str, current_train_aug_path: str) -> str:
    """
    Appends new augmented spam (label=1) into a cumulative train_aug file
    under data/processed/train_sms_mistral_augmented.csv, keeping unique messages.
    Returns the path to the merged CSV (string).
    """
    out_path = _processed_dir() / "train_sms_mistral_augmented.csv"

    df_base = pd.read_csv(current_train_aug_path) if os.path.exists(current_train_aug_path) else pd.DataFrame(columns=["message","label"])
    df_new  = pd.read_csv(new_aug_path)
    if "label" not in df_new.columns:
        df_new["label"] = 1  # generated are spam

    merged = pd.concat([df_base, df_new], ignore_index=True)
    merged = merged.drop_duplicates(subset=["message"])

    _ensure_parent(out_path)
    merged.to_csv(out_path, index=False)
    print(f"🔁 Merged augmented data → {out_path} (total {len(merged)} rows)")
    return str(out_path)

# --------------------------------------------------------------------------------------
# Summary helpers (optional; kept compatible with your previous file)
# --------------------------------------------------------------------------------------

def append_summary(iteration: int, metrics: dict, counts: dict, tau: dict, new_spam_generated: int):
    """Append one-line summary row (legacy helper; not required by run_pipeline)."""
    summary_file = _data_dir() / "mine_and_augment_summary.json"
    summary = []
    if summary_file.exists():
        with open(summary_file, "r", encoding="utf-8") as f:
            try:
                summary = json.load(f)
            except Exception:
                summary = []
    summary.append({
        "iteration": iteration,
        "accuracy": metrics.get("accuracy") if metrics else None,
        "precision": metrics.get("precision") if metrics else None,
        "recall": metrics.get("recall") if metrics else None,
        "f1": metrics.get("f1") if metrics else None,
        "tau_prob": tau.get("tau_prob") if isinstance(tau, dict) else None,
        "tau_energy": tau.get("tau_energy") if isinstance(tau, dict) else None,
        "false_negatives": counts.get("false_negatives", 0) if counts else 0,
        "borderline": counts.get("borderline", 0) if counts else 0,
        "bottom_p": counts.get("bottom_p", 0) if counts else 0,
        "new_spam_generated": new_spam_generated
    })
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"📊 Summary updated → {summary_file}")
    return summary

def should_stop_early(summary: list, patience: int = 2) -> bool:
    """Legacy helper (unused by main loop). Kept for compatibility."""
    if len(summary) <= patience:
        return False
    best_so_far = -1.0
    drops = 0
    for row in summary:
        f1 = float(row.get("f1") or 0.0)
        if f1 > best_so_far + 1e-6:
            best_so_far = f1
            drops = 0
        else:
            drops += 1
        if drops >= patience:
            return True
    return False

# --------------------------------------------------------------------------------------
# Pipeline (with sane early-stop gating)
# --------------------------------------------------------------------------------------

def run_pipeline(iters: int, per_iter_new: int, backbone: str, train_base: str, train_aug: str,
                 patience: int, min_iters: int = 2, no_early_stop: bool = False,
                 reset_summary: bool = False, exp_id: str | None = None):

    data_dir = _data_dir()
    summary_path = data_dir / "mine_and_augment_summary.json"

    # Start a fresh per-run state; do NOT poison with old global best unless resuming intentionally.
    state = {
        "exp_id": exp_id or time.strftime("%Y%m%d-%H%M%S"),
        "backbone": backbone,
        "global_best_f1_this_run": None,
        "no_improve": 0,
        "history": [],
    }

    if summary_path.exists() and not reset_summary:
        try:
            prev = json.loads(summary_path.read_text(encoding="utf-8"))
            if prev.get("backbone") == backbone and prev.get("exp_id"):
                state["exp_id"] = prev["exp_id"]
                print(f"ℹ️ Continuing with exp_id={state['exp_id']} (fresh best this run).")
        except Exception:
            pass
    else:
        print("ℹ️ Resetting summary (fresh run).")

    _ensure_parent(summary_path)
    summary_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    # Main loop
    for it in range(1, iters + 1):
        print(f"\n=== Iteration {it}/{iters} ===")

        # 1) Train + eval + mine hards (inside run_discriminator)
        current_val_f1, counts = run_discriminator(
            backbone=backbone, iteration=it,
            train_base=train_base, train_aug=train_aug
        )

        # 2) (Optional hook) mining already done
        mine_hard_examples()

        # 3) Generate augmented spam
        gen_file = run_generator(it, per_iter_new, mode="hard")

        # 4) Merge → next train_aug path
        train_aug = merge_augmented_into_train(gen_file, train_aug)

        # 5) Update run-local early-stop state
        state["history"].append({"iter": it, "f1": current_val_f1})
        gb = state["global_best_f1_this_run"]
        if gb is None or current_val_f1 > gb:
            state["global_best_f1_this_run"] = current_val_f1
            state["no_improve"] = 0
            note = f"new best → {current_val_f1:.4f}"
        else:
            state["no_improve"] += 1
            note = f"no improve ({state['no_improve']}/{patience})"

        print(f"▶ Iter {it}: F1={current_val_f1:.4f} | {note}")
        summary_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

        # 6) Early stop only after min_iters and only within this run
        if (not no_early_stop) and (it >= min_iters) and (state["no_improve"] >= patience):
            print(f"⛳ Early stopping at iteration {it} "
                  f"(no F1 improvement for {patience} iterations within this run).")
            break

    print(f"📊 Summary updated → {summary_path}")

# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=4)
    parser.add_argument("--per_iter_new", type=int, default=96)
    parser.add_argument("--backbone", type=str, default="../../models/deberta_v3-base")
    parser.add_argument("--train_base", type=str, default="../../data/processed/train_spam_merged_dedup.csv")
    parser.add_argument("--train_aug",  type=str, default="../../data/processed/train_spam_merged_dedup.csv")
    parser.add_argument("--patience", type=int, default=2)

    # New flags
    parser.add_argument("--min_iters", type=int, default=2,
                        help="Run at least this many iterations before early-stop allowed.")
    parser.add_argument("--no_early_stop", action="store_true",
                        help="Disable pipeline-level early stopping.")
    parser.add_argument("--reset_summary", action="store_true",
                        help="Ignore previous summary (fresh run).")
    parser.add_argument("--exp_id", type=str, default=None,
                        help="Optional experiment id to tag the run.")

    args = parser.parse_args()

    # Normalize paths relative to src/model
    backbone    = args.backbone
    train_base  = args.train_base
    train_aug   = args.train_aug

    run_pipeline(
        iters=args.iters,
        per_iter_new=args.per_iter_new,
        patience=args.patience,
        backbone=backbone,
        train_base=train_base,
        train_aug=train_aug,
        min_iters=args.min_iters,
        no_early_stop=args.no_early_stop,
        reset_summary=args.reset_summary,
        exp_id=args.exp_id,
    )

if __name__ == "__main__":
    main()
