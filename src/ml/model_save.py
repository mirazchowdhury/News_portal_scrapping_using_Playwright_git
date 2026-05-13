import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import BertModel, BertTokenizer, Trainer, TrainingArguments, TrainerCallback
from datasets import Dataset


# --- ১. কাস্টম মডেল আর্কিটেকচার ---
class NewsBert(nn.Module):
    def __init__(self, model_name):
        super(NewsBert, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(768, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None, **kwargs):
        out = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        x = self.fc1(out[1])
        x = self.relu(x)
        x = self.fc2(self.dropout(x))

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(x.view(-1, 2), labels.view(-1))
        return (loss, x) if loss is not None else x


# --- ২. কনফিগারেশন এবং হেল্পার ---
MODEL_NAME = "sagorsarker/bangla-bert-base"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BEST_MODEL_PATH = os.path.join(BASE_DIR, "models", "best_fake_news_model")
best_f1_score = 0.0

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n🚀 ট্রেনিং ডিভাইস: {device.upper()}")


def print_border(text):
    print("\n" + "=" * 60 + f"\n{text}\n" + "=" * 60)


# ✅ মেট্রিক্স ফাংশনটি আপডেট করা হয়েছে (Precision ও Recall সহ)
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # weighted average ব্যবহার করা হয়েছে যেহেতু ডাটা ইমব্যালেন্সড থাকতে পারে
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1_weighted': f1,
        'precision': precision,
        'recall': recall
    }


class VisualLogCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            print(
                f"📊 Epoch {state.epoch}: Loss={metrics.get('eval_loss', 0):.4f}, Acc={metrics.get('eval_accuracy', 0):.4f}")


# --- ৩. ডেটা লোড এবং ফিচার ইঞ্জিনিয়ারিং ---
print("⌛ ডেটাসেট লোড হচ্ছে...")
df = pd.read_csv('data/prothom_alo_with_fake_data.csv')


def preprocess_text(row):
    category = str(row['category']) if pd.notnull(row['category']) else "সাধারণ"
    headline = str(row['headline']) if pd.notnull(row['headline']) else ""
    content = str(row['content']) if pd.notnull(row['content']) else ""
    return f"বিভাগ: {category} শিরোনাম: {headline} বিস্তারিত: {content}"


print("⚙️ ফিচার ইঞ্জিনিয়ারিং চলছে...")
df['text'] = df.apply(preprocess_text, axis=1)
df = df[['text', 'label']].dropna()

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)


# --- ৪. ট্রেনিং লুপ (K-Fold) ---
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold = 1
final_eval_results = {}

for train_index, val_index in kf.split(df):
    print_border(f"FOLD {fold}/5 STARTING")

    train_ds = Dataset.from_pandas(df.iloc[train_index]).map(tokenize_function, batched=True)
    val_ds = Dataset.from_pandas(df.iloc[val_index]).map(tokenize_function, batched=True)

    # ইনডেক্স এরর ফিক্স
    for ds in [train_ds, val_ds]:
        if "__index_level_0__" in ds.column_names:
            ds = ds.remove_columns(["__index_level_0__"])

    model = NewsBert(MODEL_NAME).to(device)

    training_args = TrainingArguments(
        output_dir=os.path.join(BEST_MODEL_PATH, f"temp_fold_{fold}"),
        eval_strategy="epoch",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        use_cpu=False,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        remove_unused_columns=False,
        save_total_limit=1
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[VisualLogCallback()]
    )

    trainer.train()
    eval_results = trainer.evaluate()

    current_f1 = eval_results['eval_f1_weighted']

    if current_f1 > best_f1_score:
        best_f1_score = current_f1
        final_eval_results = eval_results  # সেরা মেট্রিক্স জমা রাখা
        print_border(f"⭐ NEW BEST MODEL FOUND (Fold {fold})")

        if not os.path.exists(BEST_MODEL_PATH):
            os.makedirs(BEST_MODEL_PATH)

        torch.save(model.state_dict(), os.path.join(BEST_MODEL_PATH, "pytorch_model.bin"))
        tokenizer.save_pretrained(BEST_MODEL_PATH)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    fold += 1

# --- ৫. ফাইনাল মেট্রিক্স সেভ করা (Streamlit এর জন্য) ---
if final_eval_results:
    metrics_data = {
        "accuracy": float(final_eval_results.get("eval_accuracy", 0)),
        "f1_weighted": float(final_eval_results.get("eval_f1_weighted", 0)),
        "precision": float(final_eval_results.get("eval_precision", 0)),
        "recall": float(final_eval_results.get("eval_recall", 0)),
        "loss": float(final_eval_results.get("eval_loss", 0)),
        "model_name": MODEL_NAME,
        "device": device
    }

    with open(os.path.join(BEST_MODEL_PATH, "metrics.json"), "w") as f:
        json.dump(metrics_data, f, indent=4)
    print(f"\n✅ All metrics saved successfully to metrics.json")

print_border(f"🏆 Best F1: {best_f1_score:.4f} | Path: {BEST_MODEL_PATH}")