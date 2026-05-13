import torch
import torch.nn as nn
import pandas as pd
import os
from transformers import BertModel, BertTokenizer
from tqdm import tqdm


# --- ১. ট্রেনিংয়ের মতো হুবহু একই কাস্টম ক্লাস এখানেও থাকতে হবে ---
class NewsBert(nn.Module):
    def __init__(self, model_name):
        super(NewsBert, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(768, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        x = self.fc1(out[1])
        x = self.relu(x)
        x = self.fc2(self.dropout(x))
        return x


def verify_scraped_data():
    # পাথ কনফিগারেশন
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    MODEL_PATH = os.path.join(BASE_DIR, "models", "best_fake_news_model")
    RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw_news_data.csv")
    OUTPUT_PATH = os.path.join(BASE_DIR, "data", "prothom_alo_verified_master_data.csv")

    MODEL_NAME = "sagorsarker/bangla-bert-base"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"🔍 মডেল লোড হচ্ছে এই ডিভাইস থেকে: {device}")

    try:
        # ২. মডেল এবং টোকেনাইজার লোড করা
        tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)  # সেভ করা টোকেনাইজার
        model = NewsBert(MODEL_NAME)

        # আপনার সেভ করা pytorch_model.bin লোড করা
        model.load_state_dict(torch.load(os.path.join(MODEL_PATH, "pytorch_model.bin"), map_location=device))
        model.to(device)
        model.eval()

        # ৩. ডেটা লোড এবং প্রসেসিং
        df = pd.read_csv(RAW_DATA_PATH, encoding='utf-8-sig')

        # ভেরিফিকেশন লিস্ট
        results = []

        for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="📊 ভেরিফাই করা হচ্ছে"):
            # ট্রেনিংয়ের সময় যে ফরম্যাটে ইনপুট দিয়েছিলেন, এখানেও ঠিক তাই দিতে হবে
            category = str(row['Category']) if pd.notnull(row['Category']) else "সাধারণ"
            headline = str(row['Title']) if pd.notnull(row['Title']) else ""
            content = str(row['Body']) if pd.notnull(row['Body']) else ""

            combined_text = f"বিভাগ: {category} শিরোনাম: {headline} বিস্তারিত: {content}"

            # ৪. প্রেডিকশন লজিক
            inputs = tokenizer(combined_text, return_tensors="pt", padding=True, truncation=True, max_length=256).to(
                device)

            with torch.no_grad():
                outputs = model(**inputs)
                prediction = torch.argmax(outputs, dim=1).item()

            # রেজাল্ট অ্যাসাইন করা (১ = Real, ০ = Fake)
            results.append("Real" if prediction == 1 else "Fake")

        # ৫. নতুন কলাম যুক্ত করে সেভ করা
        df['Verification'] = results
        df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
        print(f"✅ ভেরিফিকেশন সম্পন্ন! ফাইল এখানে পাবেন: {OUTPUT_PATH}")

    except Exception as e:
        print(f"❌ এরর: {str(e)}")


if __name__ == "__main__":
    verify_scraped_data()