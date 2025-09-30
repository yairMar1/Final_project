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
import torch
print(torch.version.cuda)
# --- Configuration ---
PROCESSED_DATA_DIR = "../../data/processed/"

# Data files for Stage 1
AUGMENTED_TRAIN_FILE = "train_sms_augmented.csv"
# Data files for Stage 2
AUGMENTED_MISTRAL_TRAIN_FILE = "train_sms_mistral_augmented.csv"

TEST_FILE = "test_sms.csv"  # Test file remains the same for final evaluation

# Model Output Directories
MODEL_OUTPUT_DIR_STAGE1 = "../../models/discriminator_roberta_augmented_weighted/"
MODEL_OUTPUT_DIR_STAGE2 = "../../models/discriminator_mistral_augmented/"
LOGGING_BASE_DIR = '../../logs/'  # Base directory for logs

# Hyperparameters
ROBERTA_MODEL_NAME = "roberta-base"
MAX_LENGTH = 512
RANDOM_SEED = 42
PER_DEVICE_TRAIN_BATCH_SIZE = 16
PER_DEVICE_EVAL_BATCH_SIZE = 64
NUM_TRAIN_EPOCHS = 5  # Early Stopping will likely stop it sooner

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

        # Retrieve class weights. They should now be a list/array in model.config.
        # Convert to tensor here for nn.CrossEntropyLoss
        class_weights_tensor_for_loss = torch.tensor(model.config.class_weights, dtype=torch.float).to(labels.device)

        loss_fct = nn.CrossEntropyLoss(weight=class_weights_tensor_for_loss)
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


# --- Main Training Function ---
def train_and_evaluate_model(
        model,
        tokenizer,
        train_df,
        val_df,
        test_df,
        output_dir,
        logging_subdir,
        stage_name,
        load_model_path=None  # If provided, load model from here instead of initializing new
):
    print(f"\n--- Starting {stage_name} ---")

    # Prepare and Split Data (if not already split in separate DFs)
    train_texts = train_df['message'].tolist()
    train_labels = train_df['label'].tolist()

    val_texts = val_df['message'].tolist()
    val_labels = val_df['label'].tolist()

    print(f"{stage_name} Data split: {len(train_texts)} train, {len(val_texts)} validation, {len(test_df)} test.")

    # Calculate class weights for the current training data
    print(f"\nCalculating class weights for {stage_name} to handle imbalance...")
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    # Keep class_weights as a NumPy array (or list) for JSON serialization
    # The actual tensor for loss calculation will be created in WeightedTrainer.compute_loss
    print(f"Computed class weights for {stage_name}: [Ham, Spam] -> {class_weights}")
    print(
        f"A mistake on a 'spam' sample will be penalized ~{class_weights[1] / class_weights[0]:.2f} times more than a mistake on a 'ham' sample.\n")

    # Load or Initialize Model
    if load_model_path:
        print(f"Loading model from {load_model_path} for further fine-tuning...")
        model = AutoModelForSequenceClassification.from_pretrained(load_model_path).to(DEVICE)
        tokenizer = AutoTokenizer.from_pretrained(load_model_path)  # Also load tokenizer with model
    else:
        print(f"Initializing new model: {ROBERTA_MODEL_NAME}")
        model = AutoModelForSequenceClassification.from_pretrained(
            ROBERTA_MODEL_NAME, num_labels=2
        ).to(DEVICE)

    # Assign weights to model config (as a list/array for JSON serialization)
    model.config.class_weights = class_weights.tolist()  # <--- CHANGED HERE: convert to list for JSON serialization

    # Tokenize Datasets
    print(f"Tokenizing datasets for {stage_name}...")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=MAX_LENGTH)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=MAX_LENGTH)
    test_encodings = tokenizer(test_df['message'].tolist(), truncation=True, padding=True, max_length=MAX_LENGTH)

    train_dataset = SMSDataset(train_encodings, train_labels)
    val_dataset = SMSDataset(val_encodings, val_labels)
    test_dataset = SMSDataset(test_encodings, test_df['label'].tolist())

    # Training Arguments
    logging_dir = os.path.join(LOGGING_BASE_DIR, logging_subdir)
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
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    # Train
    print(f"\nStarting {stage_name} Training...")
    trainer.train()

    # Final Evaluation
    print(f"\n{stage_name} complete. Evaluating the best model on the held-out test set...")
    test_results = trainer.evaluate(eval_dataset=test_dataset)

    print("\n" + "#" * 50)
    print(f"##         FINAL TEST SET RESULTS ({stage_name})        ##")
    print("#" * 50)
    for key, value in test_results.items():
        if isinstance(value, float): print(f"  {key.replace('eval_', '').capitalize():<12}: {value:.4f}")
    print("#" * 50)

    # Save Model
    best_model_path = os.path.join(output_dir, "best_model")
    trainer.save_model(best_model_path)
    tokenizer.save_pretrained(best_model_path)
    print(f"\nBest {stage_name} model saved to {best_model_path}")

    return model, tokenizer, best_model_path


if __name__ == "__main__":
    # --- Load Test Data Once ---
    test_path = os.path.join(PROCESSED_DATA_DIR, TEST_FILE)
    df_test = load_data(test_path)
    if df_test is None:
        print("Exiting due to test data loading errors.")
        exit()

    # --- Stage 1: Train on AUGMENTED_TRAIN_FILE ---
    augmented_train_path = os.path.join(PROCESSED_DATA_DIR, AUGMENTED_TRAIN_FILE)
    df_train_val_stage1 = load_data(augmented_train_path)
    if df_train_val_stage1 is None:
        print("Exiting due to Stage 1 train data loading errors.")
        exit()

    # Split train_val for Stage 1
    train_df_stage1, val_df_stage1 = train_test_split(
        df_train_val_stage1, test_size=0.1, random_state=RANDOM_SEED, stratify=df_train_val_stage1['label']
    )

    # Initialize tokenizer for the first stage
    initial_tokenizer = AutoTokenizer.from_pretrained(ROBERTA_MODEL_NAME)

    # Run Stage 1 Training
    _, _, stage1_model_path = train_and_evaluate_model(
        model=None,  # Will be initialized from ROBERTA_MODEL_NAME
        tokenizer=initial_tokenizer,
        train_df=train_df_stage1,
        val_df=val_df_stage1,
        test_df=df_test,
        output_dir=MODEL_OUTPUT_DIR_STAGE1,
        logging_subdir="discriminator_roberta_augmented_weighted_logs",
        stage_name="Discriminator RoBERTa Augmented Weighted"
    )

    # --- Stage 2: Further Fine-tune on AUGMENTED_MISTRAL_TRAIN_FILE ---
    augmented_mistral_train_path = os.path.join(PROCESSED_DATA_DIR, AUGMENTED_MISTRAL_TRAIN_FILE)
    df_train_val_stage2 = load_data(augmented_mistral_train_path)
    if df_train_val_stage2 is None:
        print("Exiting due to Stage 2 train data loading errors.")
        exit()

    # Split train_val for Stage 2
    train_df_stage2, val_df_stage2 = train_test_split(
        df_train_val_stage2, test_size=0.1, random_state=RANDOM_SEED, stratify=df_train_val_stage2['label']
    )

    # Run Stage 2 Training (loading the model saved from Stage 1)
    train_and_evaluate_model(
        model=None,  # The function will load the model from stage1_model_path
        tokenizer=None,  # The function will load the tokenizer from stage1_model_path
        train_df=train_df_stage2,
        val_df=val_df_stage2,
        test_df=df_test,
        output_dir=MODEL_OUTPUT_DIR_STAGE2,
        logging_subdir="discriminator_mistral_augmented_logs",
        stage_name="Discriminator Mistral Augmented",
        load_model_path=stage1_model_path  # Crucial: load the best model from stage 1
    )

    print("\nAll training stages completed successfully!")