import os
from PyPDF2 import PdfReader
import pandas as pd
from datasets import Dataset
from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments

# Directories
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
MODEL_DIR = "models/fine_tuned_bart"


def extract_text_from_pdf(pdf_path):
    """
    Extract text from a single PDF file.
    """
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    print(f"Extracted text from {pdf_path}")
    return text


def prepare_dataset():
    """
    Extract text from PDFs and prepare a summarization dataset.
    """
    # Load summaries from a CSV file
    summary_file = os.path.join(PROCESSED_DATA_DIR, "summaries.csv")
    summary_df = pd.read_csv(summary_file)

    # Extract text from PDFs and align with summaries
    pdf_texts = []
    summaries = []

    for index, row in summary_df.iterrows():
        pdf_path = os.path.join(RAW_DATA_DIR, row["filename"])
        if os.path.exists(pdf_path):
            text = extract_text_from_pdf(pdf_path)
            pdf_texts.append(text)
            summaries.append(row["summary"])
        else:
            print(f"File not found: {pdf_path}")

    # Create a dataset DataFrame
    dataset_df = pd.DataFrame({"text": pdf_texts, "summary": summaries})
    dataset_csv = os.path.join(PROCESSED_DATA_DIR, "summarization_dataset.csv")
    dataset_df.to_csv(dataset_csv, index=False)
    print(f"Dataset saved to {dataset_csv}")
    return dataset_df


def preprocess_dataset(dataset_df):
    """
    Preprocess the dataset for BART fine-tuning.
    """
    # Convert to Hugging Face Dataset
    dataset = Dataset.from_pandas(dataset_df)

    # Split into training and evaluation sets
    dataset_split = dataset.train_test_split(test_size=0.1)
    train_data = dataset_split["train"]
    eval_data = dataset_split["test"]

    # Load tokenizer
    model_name = "facebook/bart-large"
    tokenizer = BartTokenizer.from_pretrained(model_name)

    # Preprocess data
    def preprocess_data(examples):
        inputs = tokenizer(
            examples["text"],
            max_length=1024,
            truncation=True,
            padding="max_length",
        )
        labels = tokenizer(
            examples["summary"],
            max_length=128,
            truncation=True,
            padding="max_length",
        )
        inputs["labels"] = labels["input_ids"]
        return inputs

    train_data = train_data.map(preprocess_data, batched=True)
    eval_data = eval_data.map(preprocess_data, batched=True)
    print("Preprocessing complete")
    return train_data, eval_data, tokenizer


def fine_tune_bart(train_data, eval_data, tokenizer):
    """
    Fine-tune the BART model on the processed dataset.
    """
    # Load pre-trained BART model
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=MODEL_DIR,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=10,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        tokenizer=tokenizer,
    )

    # Train the model
    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    print(f"Model saved to {MODEL_DIR}")


if __name__ == "__main__":
    # Prepare dataset
    dataset_df = prepare_dataset()

    # Preprocess dataset
    train_data, eval_data, tokenizer = preprocess_dataset(dataset_df)

    # Fine-tune BART
    fine_tune_bart(train_data, eval_data, tokenizer)
