import datasets
import numpy as np
import torch.cuda
from rouge_score import rouge_scorer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer

device = "cuda" if torch.cuda.is_available() else "cpu"

# load data set
dataset = datasets.load_dataset("FiscalNote/billsum")

# load tokenizer and model
model_name = "Falconsai/text_summarization"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = [scorer.score(pred, ref) for pred, ref in zip(decoded_preds, decoded_labels)]

    rouge1 = np.mean([s["rouge1"].fmeasure for s in scores])
    rouge2 = np.mean([s["rouge2"].fmeasure for s in scores])
    rougeL = np.mean([s["rougeL"].fmeasure for s in scores])

    return {"rouge1": rouge1, "rouge2": rouge2, "rougeL": rougeL}


def process_function(context, max_input_length=512, max_out_length=128):
    inputs = context["text"]
    outputs = context["summary"]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding="max_length")
    labels = tokenizer(outputs, max_length=max_out_length, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


if __name__ == '__main__':
    tokenized_datasets = dataset.map(lambda x: process_function(x))

    train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(10000))
    eval_dateset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

    train_args = TrainingArguments(output_dir="../model", eval_strategy="epoch",
                                   per_device_train_batch_size=8, per_device_eval_batch_size=2,
                                   # gradient_accumulation_steps=8,
                                   learning_rate=3e-5, num_train_epochs=1,
                                   weight_decay=0.01,
                                   save_total_limit=3,
                                   fp16=True,
                                   fp16_full_eval=False,
                                   logging_dir="./logs",
                                   report_to="none")

    trainer = Trainer(model=model, args=train_args, train_dataset=train_dataset, eval_dataset=eval_dateset,
                      compute_metrics=compute_metrics)

    trainer.train()
    model.save_pretrained("../model/trained_model")
    tokenizer.save_pretrained("../model/trained_tokenizer")
