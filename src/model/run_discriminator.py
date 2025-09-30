

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, \
    EarlyStoppingCallback, IntervalStrategy
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
import numpy as np
import os
import shutil
import argparse

# --- Configuration ---
PROCESSED_DATA_DIR = "../../data/processed/"

# 👇 Edit these two lines and just press Run:
TRAIN_FILE = "train_sms_mistral_augmented.csv"
MODEL_NAME = "discriminator_mistral_aug"

TEST_FILE = "test_sms.csv"

# Base directories for models and logs
BASE_MODEL_OUTPUT_DIR = "../../models/discriminator_single_run/"
BASE_LOGGING_DIR = '../../logs/discriminator_single_run/'

# Hyperparameters
ROBERTA_MODEL_NAME = "roberta-base"
MAX_LENGTH = 512
RANDOM_SEED = 42
PER_DEVICE_TRAIN_BATCH_SIZE = 16
PER_DEVICE_EVAL_BATCH_SIZE = 64
NUM_TRAIN_EPOCHS = 5

# --- Device Setup (Moved to top for immediate feedback) ---
print(f"Is CUDA available? {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    DEVICE = torch.device("cuda")
else:
    print("CUDA is NOT available. This may impact performance significantly if you have a GPU.")
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")

# --- Helper Functions and Classes ---

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        df['message'] = df['message'].fillna('').astype(str)
        df['label'] = df['label'].astype(int)
        print(f"Successfully loaded data from {file_path}, shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None

class SMSDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', pos_label=1,
                                                               zero_division=0)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # Ensure class_weights_tensor_for_loss is on the correct device
        class_weights_tensor_for_loss = torch.tensor(model.config.class_weights, dtype=torch.float).to(labels.device)
        loss_fct = nn.CrossEntropyLoss(weight=class_weights_tensor_for_loss)
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# --- Main Training Function ---
def train_single_discriminator(
        train_data_file: str,
        test_df: pd.DataFrame,
        output_name: str
):
    print(f"\n--- Starting Discriminator Training for: {train_data_file} ---")

    train_file_path = os.path.join(PROCESSED_DATA_DIR, train_data_file)
    df_train_val = load_data(train_file_path)
    if df_train_val is None:
        print(f"Exiting due to data loading errors for {train_data_file}.")
        return

    train_df, val_df = train_test_split(
        df_train_val, test_size=0.1, random_state=RANDOM_SEED, stratify=df_train_val['label']
    )

    print(f"Data split: {len(train_df)} train, {len(val_df)} validation, {len(test_df)} test.")

    print(f"\nCalculating class weights for {output_name} to handle imbalance...")
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_df['label'].tolist()),
        y=train_df['label'].tolist()
    )
    print(f"Computed class weights for {output_name}: [Ham, Spam] -> {class_weights}")
    print(
        f"A mistake on a 'spam' sample will be penalized ~{class_weights[1] / class_weights[0]:.2f} times more than a mistake on a 'ham' sample.\n")

    print(f"Initializing new model: {ROBERTA_MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(
        ROBERTA_MODEL_NAME, num_labels=2
    ).to(DEVICE) # Ensure model is moved to the selected device
    tokenizer = AutoTokenizer.from_pretrained(ROBERTA_MODEL_NAME)

    model.config.class_weights = class_weights.tolist()

    print(f"Tokenizing datasets for {output_name}...")
    train_encodings = tokenizer(train_df['message'].tolist(), truncation=True, padding=True, max_length=MAX_LENGTH)
    val_encodings = tokenizer(val_df['message'].tolist(), truncation=True, padding=True, max_length=MAX_LENGTH)
    test_encodings = tokenizer(test_df['message'].tolist(), truncation=True, padding=True, max_length=MAX_LENGTH)

    train_dataset = SMSDataset(train_encodings, train_df['label'].tolist())
    val_dataset = SMSDataset(val_encodings, val_df['label'].tolist())
    test_dataset = SMSDataset(test_encodings, test_df['label'].tolist())

    output_dir = os.path.join(BASE_MODEL_OUTPUT_DIR, output_name)
    logging_dir = os.path.join(BASE_LOGGING_DIR, output_name + "_logs")

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(logging_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_dir=logging_dir,
        logging_steps=100,
        eval_strategy=IntervalStrategy.EPOCH,
        save_strategy=IntervalStrategy.EPOCH,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
        seed=RANDOM_SEED,
        # Ensure data is moved to the correct device
        no_cuda=False if DEVICE.type == 'cuda' else True # This will be set by default based on torch.cuda.is_available()
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    print(f"\nStarting Training for {output_name}...")
    trainer.train()

    print(f"\nTraining for {output_name} complete. Evaluating the best model on the held-out test set...")
    test_results = trainer.evaluate(eval_dataset=test_dataset)

    print("\n" + "#" * 50)
    print(f"##  FINAL TEST SET RESULTS ({output_name})        ##")
    print("#" * 50)
    for key, value in test_results.items():
        if isinstance(value, float): print(f"  {key.replace('eval_', '').capitalize():<12}: {value:.4f}")
    print("#" * 50)

    best_model_path = os.path.join(output_dir, "best_model")
    trainer.save_model(best_model_path)
    tokenizer.save_pretrained(best_model_path)
    print(f"\nBest model for {output_name} saved to {best_model_path}")

    return test_results


def main():
    # --- Load Test Data Once ---
    test_path = os.path.join(PROCESSED_DATA_DIR, TEST_FILE)
    df_test = load_data(test_path)
    if df_test is None:
        print("Exiting due to test data loading errors.")
        exit()

    # Run the single discriminator training
    train_single_discriminator(
        train_data_file=TRAIN_FILE,
        test_df=df_test,
        output_name=MODEL_NAME
    )

    print(f"\n✅ Discriminator training on '{TRAIN_FILE}' completed successfully!")
    print(f"📁 Model saved under: {os.path.join(BASE_MODEL_OUTPUT_DIR, MODEL_NAME)}")

if __name__ == "__main__":
    main()

