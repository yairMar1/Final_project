import os, argparse, pandas as pd, random, re
from pathlib import Path
from transformers import pipeline

# ── Paths (robust, relative to this file) ─────────────────────
FILE_DIR = Path(__file__).resolve().parent           # .../src/model
PROJ_ROOT = FILE_DIR.parents[1]                      # .../ (שורש הפרויקט)
DATA_DIR = PROJ_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

DEFAULT_OUTPUT = str(DATA_DIR / "augmented_spam_v2.csv")
TRAIN_FILE = str(PROCESSED_DIR / "train_sms.csv")

HARD_FILES_DEFAULT = [
    "hard_spam_false_negatives.csv",
    "hard_spam_borderline.csv",
    "hard_spam_bottompct.csv",
]


# ---- Prompts ----
FULL_PROMPT = (
    "Generate a short spam SMS message. It should create urgency, offer a prize, ask to click a link or call a number. "
    "Use natural language. Return ONLY the message text, one line, no quotes, no explanations."
)

HARD_PROMPT_TEMPLATE = (
    "Create one realistic spam SMS. Similar to:\n{examples}\n"
    "Output only the message text itself — nothing else."
)





# ---- Utils ----
def _clean_line(s: str) -> str:
    s = s.strip().replace("\n", " ")
    s = re.sub(r"^['\"`]|['\"`]$", "", s).strip()
    s = re.sub(r"\s+", " ", s)
    return s

def _normalize_for_dedup(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" '\"`")
    return s

def _valid_spam(s: str) -> bool:
    s = s.strip()
    if len(s) < 8:              # shorter than 8 chars → too short
        return False
    if len(s.split()) < 2:      # fewer than 2 words → too short
        return False
    # block obvious prompt leakage, but keep it lightweight
    bad_snippets = ["Examples:", "You are an expert", "Return ONLY"]
    if any(b.lower() in s.lower() for b in bad_snippets):
        return False
    return True


def load_hard_examples_from_processed(hard_files=None, min_len=6, drop_dups=True):
    if hard_files is None:
        hard_files = HARD_FILES_DEFAULT
    dfs = []
    for raw in hard_files:
        p = Path(raw.strip())
        # אם לא נתיב מוחלט ולא קיים ככה, חפש תחת processed/
        if not p.is_absolute() and not p.exists():
            p = PROCESSED_DIR / p.name
        if not p.exists():
            print(f"ℹ️ missing: {p}")
            continue
        df = pd.read_csv(p)
        if "message" not in df.columns:
            print(f"⚠️ no 'message' column in {p} – skipping.")
            continue
        dfs.append(df[["message"]].copy())

    if not dfs:
        print("❌ No hard files found under processed/")
        return []

    hard = pd.concat(dfs, ignore_index=True)
    hard["message"] = hard["message"].astype(str).str.strip()
    hard = hard[hard["message"].str.len() >= min_len]

    if drop_dups:
        hard["_k"] = hard["message"].map(_normalize_for_dedup)
        hard = hard.drop_duplicates(subset=["_k"]).drop(columns=["_k"])

    msgs = hard["message"].tolist()
    print(f"✅ Hard pool loaded: {len(msgs)} rows (after clean/dedup) from {len(dfs)} file(s)")
    return msgs

def sample_real_spam(n=3):
    if not os.path.exists(TRAIN_FILE):
        return []
    df = pd.read_csv(TRAIN_FILE)
    df = df[df["label"] == 1]
    if df.empty: return []
    return random.sample(df["message"].tolist(), min(n, len(df)))

def load_train_norm_set():
    if not os.path.exists(TRAIN_FILE):
        return set()
    df = pd.read_csv(TRAIN_FILE)
    if "message" not in df.columns: return set()
    msgs = df["message"].astype(str).tolist()
    return { _normalize_for_dedup(m) for m in msgs }

# ---- Core gen ----
def generate_spam_messages(generator, num_messages=500, mode="full", hard_examples=None,
                           per_prompt=3, batch_size=16, dedup_against_train=False):
    """
    per_prompt: כמה דוגמאות לייצר מכל prompt
    batch_size: כמה prompts להריץ במקביל (בקירוב)
    """
    # diagnostics
    reasons = {"too_short":0, "invalid":0, "duplicate_in_batch":0, "duplicate_in_train":0}
    accepted = []
    seen = set()
    train_norm_set = load_train_norm_set() if dedup_against_train else set()

    # חלק את העבודה לפרומפטים
    prompts = []
    if mode == "full":
        real_spam_examples = sample_real_spam(3)
        examples_text = "\n".join(f"- {m}" for m in real_spam_examples)
        base_prompt = FULL_PROMPT + "\n\nInspiration:\n" + examples_text
        prompts = [base_prompt] * ((num_messages + per_prompt - 1) // per_prompt)

    elif mode == "hard":
        if not hard_examples:
            raise ValueError("Hard mode requires a list of hard examples.")
        # בנה prompts מתוך דוגמאות hard שונות
        pool = hard_examples[:]
        random.shuffle(pool)
        # כל פרומפט משתמש ב-3 דוגמאות אקראיות
        for i in range(0, len(pool), 3):
            few = pool[i:i+3]
            if not few: break
            examples_str = "\n".join(f"- {m}" for m in few)
            prompts.append(HARD_PROMPT_TEMPLATE.format(examples=examples_str))
        if not prompts:
            prompts = [HARD_PROMPT_TEMPLATE.format(examples="\n".join(f"- {m}" for m in random.sample(hard_examples, min(3, len(hard_examples)))))]

    # ודא שהטוקנייזר יודע pad
    generator.tokenizer.pad_token = generator.tokenizer.eos_token

    # הרצה בבאצ'ים
    target = num_messages
    idx = 0
    while len(accepted) < target and idx < len(prompts):
        batch_prompts = prompts[idx: idx + batch_size]
        idx += batch_size

        bad_words = ["sample message", "example", "examples:", "instruction", "return only", "output:", "spam"]
        # build bad-words ids from the generator's tokenizer (avoid using undefined `gen`)
        bad_words_ids = generator.tokenizer(bad_words, add_special_tokens=False).input_ids

        outputs = generator(
            batch_prompts,
            max_new_tokens=48,
            do_sample=True,
            temperature=0.7,  # קצת יותר יציב
            top_p=0.95,
            repetition_penalty=1.1,
            num_return_sequences=per_prompt,
            pad_token_id=generator.tokenizer.eos_token_id,
            bad_words_ids=bad_words_ids,  # ← חוסם מילים לא רצויות
        )

        # flatten safely: pipeline may return list[list[dict]] if batched+num_return_sequences>1
        if outputs and isinstance(outputs[0], list):
            flat = (d for sub in outputs for d in sub)
        else:
            flat = iter(outputs)

        for d in flat:
            # handle both 'generated_text' (text-generation) and 'text' (edge cases)
            text = d.get("generated_text", d.get("text", ""))
            s = _clean_line(text)

            if not _valid_spam(s):
                reasons["invalid"] += 1
                continue

            norm = _normalize_for_dedup(s)
            if norm in seen:
                reasons["duplicate_in_batch"] += 1
                continue
            if norm in train_norm_set:
                reasons["duplicate_in_train"] += 1
                continue

            seen.add(norm)
            accepted.append(s)
            if len(accepted) >= target:
                break

    print(f"raw kept: {len(accepted)} / target {target} | drop reasons={reasons}")
    # דדופ סופי
    accepted = list(dict.fromkeys(accepted))
    return accepted

# ---- CLI ----
def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, default="full", choices=["full", "hard"])
    ap.add_argument("--num_messages", type=int, default=None)
    ap.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    ap.add_argument("--hard_files", type=str, default="")  # comma-separated under processed/
    ap.add_argument("--per_prompt", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--dedup_train", action="store_true")
    return ap

def main():
    args = build_argparser().parse_args()
    if args.num_messages is None:
        args.num_messages = 2000 if args.mode == "full" else 500

    # טען hard
    hard_examples = None
    if args.mode == "hard":
        files_list = None
        if args.hard_files.strip():
            files_list = [s.strip() for s in args.hard_files.split(",") if s.strip()]
        hard_examples = load_hard_examples_from_processed(files_list)
        print("Incoming hard examples actually used:", len(hard_examples))
        if not hard_examples:
            pd.DataFrame(columns=["message", "label"]).to_csv(args.output, index=False)
            print(f"✅ Generated 0 messages in hard mode → saved to {args.output}")
            return

    gen = pipeline("text-generation",
                   model="mistralai/Mistral-7B-Instruct-v0.2",
                   device=0,
                   return_full_text=False)  # ← חשוב
    gen.tokenizer.pad_token = gen.tokenizer.eos_token

    messages = generate_spam_messages(
        gen,
        num_messages=args.num_messages,
        mode=args.mode,
        hard_examples=hard_examples,
        per_prompt=args.per_prompt,
        batch_size=args.batch_size,
        dedup_against_train=args.dedup_train,
    )

    df_out = pd.DataFrame({"message": messages, "label": [1] * len(messages)})
    df_out.to_csv(args.output, index=False)
    print(f"✅ Generated {len(messages)} messages in {args.mode} mode → saved to {args.output}")

if __name__ == "__main__":
    main()
