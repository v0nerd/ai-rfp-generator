import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
import torch
from torch.nn import BCEWithLogitsLoss

MODEL_DIR = "models/fine_tuned_bart"


def prepare_multi_label_dataset(csv_path="data/processed/compliance_dataset.csv"):
    """
    Load and prepare a multi-label dataset from a CSV file.
    """
    # Load dataset from CSV
    df = pd.read_csv(csv_path)

    # Convert to Hugging Face Dataset
    dataset = Dataset.from_pandas(df)

    # Split into train and validation datasets
    train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
    return train_test_split


def preprocess_function(examples, tokenizer):
    """
    Tokenize input text using the BERT tokenizer.
    """
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
    )


# Custom Trainer for Multi-Label Classification
class MultiLabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Custom loss function for multi-label classification using BCEWithLogitsLoss.
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = BCEWithLogitsLoss()
        loss = loss_fct(logits, labels.float())
        return (loss, outputs) if return_outputs else loss


def fine_tune_multi_label_bert(csv_path="data/processed/compliance_dataset.csv"):
    """
    Fine-tune BERT for multi-label compliance classification using a CSV dataset.
    """
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)  # 3 labels for timeline, budget, privacy

    # Prepare dataset
    dataset = prepare_multi_label_dataset(csv_path)
    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer), batched=True
    )

    # Add labels as tensors
    def format_labels(examples):
        labels = ["timeline", "budget", "privacy"]
        examples["labels"] = torch.tensor([examples[label] for label in labels])  # Ensure correct shape
        return examples

    tokenized_dataset = tokenized_dataset.map(format_labels)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=MODEL_DIR,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy="epoch",
        logging_dir="./logs",
    )

    # Use the custom MultiLabelTrainer
    trainer = MultiLabelTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
    )

    # Train the model
    trainer.train()

    # Save the model
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    print("Model fine-tuned and saved to models/fine_tuned_bert")


if __name__ == "__main__":
    fine_tune_multi_label_bert()
