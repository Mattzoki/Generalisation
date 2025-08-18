import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
from transformers import (
    set_seed, Trainer, TrainingArguments,
    AutoTokenizer, AutoModelForTokenClassification,
    DataCollatorForTokenClassification
)
from datasets import load_from_disk
import evaluate

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
conll_label_to_id = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3,
                     'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}
id2label = {v: k for k, v in conll_label_to_id.items()}

metric = evaluate.load("seqeval")

conll_encoded = load_from_disk("./splits/conll_encoded")
ontonotes_encoded = load_from_disk("./splits/ontonotes_encoded")

data_collator = DataCollatorForTokenClassification(tokenizer)


def compute_metrics(p):
    predictions, labels = p
    predictions = predictions.argmax(axis=-1)

    true_predictions = [
        [id2label[pred]
            for pred, label in zip(prediction, label_row) if label != -100]
        for prediction, label_row in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[label]
            for pred, label in zip(prediction, label_row) if label != -100]
        for prediction, label_row in zip(predictions, labels)
    ]

    return metric.compute(predictions=true_predictions, references=true_labels)


if __name__ == "__main__":
    seeds = [42, 100, 2023, 7, 999]
    results = []

    for seed in seeds:
        print(f"Running for seed {seed}")
        set_seed(seed)

        model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=len(conll_label_to_id)
        )

        args = TrainingArguments(
            output_dir=f"./results/conll_seed_{seed}",
            seed=seed,
            eval_strategy="no",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            gradient_accumulation_steps=2,
            num_train_epochs=3,
            weight_decay=0.01,
            load_best_model_at_end=True,
            report_to="none",
            fp16=True,
            logging_steps=1000,
            save_strategy="no",
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=conll_encoded,
            eval_dataset=ontonotes_encoded,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        eval_result = trainer.evaluate(ontonotes_encoded)
        eval_result['seed'] = seed
        results.append(eval_result)

    for r in results:
        print(f"Seed: {r['seed']}, F1: {r['eval_f1']:.4f}, Accuracy: {r['eval_accuracy']:.4f}")

    df = pd.DataFrame(results)

    plt.figure(figsize=(10, 5))
    sns.set(style="whitegrid")

    metrics_to_plot = ['eval_f1', 'eval_accuracy']

    for metric in metrics_to_plot:
        sns.lineplot(x='seed', y=metric, data=df, marker='o', label=metric)

    plt.title("Evaluation Metrics across Random Seeds")
    plt.xlabel("Seed")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.show()
