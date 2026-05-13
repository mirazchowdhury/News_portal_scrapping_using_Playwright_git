import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from datasets import Dataset

# ১. সেরা মডেল এবং টোকেনাইজার লোড করা
MODEL_PATH = "./best_fake_news_model"
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH).to("cuda")

# ২. টেস্ট করার জন্য ডেটা লোড করা
df = pd.read_csv('prothom_alo_with_fake_data.csv')
df['text'] = df['headline'].astype(str) + " " + df['content'].astype(str)
test_df = df.sample(n=500, random_state=42) # ৫০০টি র‍্যান্ডম ডেটা দিয়ে টেস্ট

# ৩. প্রেডিকশন ফাংশন
def get_predictions(text_list):
    inputs = tokenizer(text_list, padding=True, truncation=True, max_length=128, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)
    return preds.cpu().numpy()

# ব্যাচ অনুযায়ী প্রেডিকশন নেওয়া
print("🔍 প্রেডিকশন নেওয়া হচ্ছে...")
y_true = test_df['label'].values
y_pred = get_predictions(test_df['text'].tolist())

# ৪. Confusion Matrix তৈরি
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Fake (0)', 'Real (1)'],
            yticklabels=['Fake (0)', 'Real (1)'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - Fake News Detection')
plt.show()

# ৫. বিস্তারিত রিপোর্ট (Precision, Recall, F1)
print("\n📊 Detailed Classification Report:")
print(classification_report(y_true, y_pred, target_names=['Fake', 'Real']))