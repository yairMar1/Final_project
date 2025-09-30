import torch
from transformers import pipeline, set_seed, AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import os
import random

# --- Configuration ---

## <<< שינוי: הגדרת נתיב מקומי למודל במקום שם מה-Hub
# הנתיב הוא יחסי למיקום הסקריפט (src/model)
LOCAL_MODEL_PATH = "../../models/Mistral-7B-Instruct-v0.2/"

# בדיקה שהנתיב למודל המקומי קיים
if not os.path.exists(LOCAL_MODEL_PATH):
    raise FileNotFoundError(
        f"Model directory not found at '{LOCAL_MODEL_PATH}'. "
        "Please ensure you have downloaded the model and placed it in the project's 'models' directory."
    )

# הגדרות קבצים
PROCESSED_DATA_DIR = "../../data/processed/"
TRAIN_SMS_FILE = "train_sms.csv"
LLM_AUGMENTED_TRAIN_FILE = "train_sms_mistral_augmented.csv"

# הגדרות לתהליך היצירה
NUM_SPAM_TO_GENERATE = 2000
NUM_EXAMPLES_IN_PROMPT = 5
MAX_NEW_TOKENS = 60

# הגדרות לשחזוריות
RANDOM_SEED = 42
set_seed(RANDOM_SEED)


def load_original_training_data(file_path):
    """טוען את דאטה-סט האימון המקורי ומבצע בדיקות תקינות בסיסיות."""
    try:
        df = pd.read_csv(file_path)
        if 'message' not in df.columns or 'label' not in df.columns:
            raise KeyError("File must contain 'message' and 'label' columns.")

        df['message'] = df['message'].astype(str).fillna('')
        df['label'] = df['label'].astype(int)
        print(f"Successfully loaded data from {file_path}, shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}. Please run load_preprocess.py first.")
        return None
    except Exception as e:
        print(f"An error occurred while loading data: {e}")
        return None


def build_mistral_prompt(spam_examples):
    """בונה את ההוראה (prompt) בפורמט Chat Template שמתאים למודלי Instruct כמו Mistral."""
    instruction = (
        "You are an expert spam message writer. Your task is to generate a new, short spam SMS message. "
        "The message should be similar in style and intent to the following examples. "
        "It must create a sense of urgency, offer a fake prize, or ask the user to click a link or call a number. "
        "VERY IMPORTANT: Do not write any introductory text like 'Here is the new spam message:'. "
        "Provide ONLY the new spam message text directly.\n\n"
        "Here are the examples of the style I want:\n"
    )

    for example in spam_examples:
        instruction += f"- {example}\n"

    prompt = f"<s>[INST] {instruction.strip()} [/INST]"
    return prompt


def generate_spam_messages(generator_pipeline, df_spam, num_to_generate):
    """מייצר הודעות ספאם חדשות באמצעות ה-LLM, עם לוגיקה לטיפול בשגיאות."""
    newly_generated_spam = []
    print(f"Starting LLM-based generation of {num_to_generate} new spam messages using Mistral-7B (bfloat16)...")

    spam_message_list = df_spam['message'].unique().tolist()
    if len(spam_message_list) < NUM_EXAMPLES_IN_PROMPT:
        print(
            f"Warning: Not enough unique spam examples ({len(spam_message_list)}) to build prompts. Using duplicates.")
        spam_message_list = df_spam['message'].tolist()

    for i in range(num_to_generate):
        spam_examples = random.sample(spam_message_list, NUM_EXAMPLES_IN_PROMPT)
        prompt_text = build_mistral_prompt(spam_examples)

        try:
            generated_outputs = generator_pipeline(
                prompt_text,
                max_new_tokens=MAX_NEW_TOKENS,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.75,
                top_k=50,
                top_p=0.95,
                pad_token_id=generator_pipeline.tokenizer.eos_token_id
            )

            generated_text_full = generated_outputs[0]['generated_text']
            new_message = generated_text_full.split("[/INST]")[-1].strip()
            new_message = new_message.replace("<s>", "").replace("</s>", "").strip()
            new_message_cleaned = " ".join(new_message.splitlines()).strip()

            if new_message_cleaned:
                newly_generated_spam.append(new_message_cleaned)

            if (i + 1) % 50 == 0 or (i + 1) == num_to_generate:
                print(f"Generated {i + 1}/{num_to_generate} messages...")

        except Exception as e:
            print(f"An error occurred during generation at step {i + 1}: {e}")
            continue

    print(f"Finished generation. Successfully created {len(newly_generated_spam)} new spam messages.")
    return newly_generated_spam


if __name__ == "__main__":
    train_file_path = os.path.join(PROCESSED_DATA_DIR, TRAIN_SMS_FILE)
    df_train_original = load_original_training_data(train_file_path)

    if df_train_original is not None:
        ## <<< שינוי: הודעת הלוג עודכנה כדי לשקף טעינה מקומית
        print(f"Initializing LLM pipeline from local path: {LOCAL_MODEL_PATH}...")
        try:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available. This script requires a GPU.")
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("bfloat16 precision is not supported on this GPU. Cannot proceed.")

            ## <<< שינוי: טוענים מהנתיב המקומי במקום משם המודל ב-Hub
            model = AutoModelForCausalLM.from_pretrained(
                LOCAL_MODEL_PATH,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )

            ## <<< שינוי: טוענים את הטוקנייזר גם מהנתיב המקומי
            tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            text_generator = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer
            )
            print("Pipeline initialized successfully from local model files using bfloat16 precision on GPU.")

        except Exception as e:
            print(f"Failed to initialize the pipeline. Error: {e}")
            print("Please ensure you have installed all required packages: transformers, torch, accelerate.")
            exit()

        df_original_spam = df_train_original[df_train_original['label'] == 1].copy()

        if df_original_spam.empty:
            print("No spam messages found in the training data to use as examples.")
        else:
            new_spam_list = generate_spam_messages(text_generator, df_original_spam, NUM_SPAM_TO_GENERATE)
            df_newly_generated_spam = pd.DataFrame({'label': 1, 'message': new_spam_list})
            df_final_augmented_train = pd.concat([df_train_original, df_newly_generated_spam], ignore_index=True)
            augmented_train_path = os.path.join(PROCESSED_DATA_DIR, LLM_AUGMENTED_TRAIN_FILE)
            df_final_augmented_train.to_csv(augmented_train_path, index=False)
            print(f"\nSuccessfully saved the Mistral (bfloat16) augmented dataset to: {augmented_train_path}")
            print(f"Final augmented data shape: {df_final_augmented_train.shape}")