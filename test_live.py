import torch
import torch.nn as nn
import os
from transformers import BertModel, BertTokenizer


# ১. আপনার কাস্টম মডেল ক্লাস (model_save.py এর মতোই হতে হবে)
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


def predict_news(category, headline, content):
    # পাথ সেটআপ
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "models", "best_fake_news_model")
    MODEL_NAME = "sagorsarker/bangla-bert-base"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Using Device: {device.upper()}")

    # ২. মডেল এবং টোকেনাইজার লোড
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    model = NewsBert(MODEL_NAME)

    # ওয়েট লোড করা
    model.load_state_dict(torch.load(os.path.join(MODEL_PATH, "pytorch_model.bin"), map_location=device))
    model.to(device)
    model.eval()

    # ৩. ইনপুট ফরম্যাটিং (ঠিক ট্রেনিংয়ের মতো)
    combined_text = f"বিভাগ: {category} শিরোনাম: {headline} বিস্তারিত: {content}"

    inputs = tokenizer(combined_text, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)

    # ৪. প্রেডিকশন
    with torch.no_grad():
        outputs = model(**inputs)
        # Softmax দিয়ে কনফিডেন্স বের করা
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][prediction].item() * 100

    result = "Real" if prediction == 1 else "Fake"
    print(f"\n--- Result ---")
    print(f"Prediction: {result}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"--------------\n")


if __name__ == "__main__":
    # টেস্ট করার জন্য এখানে তথ্য দিন
    test_cat = "Sports"
    test_head = "রিয়ালের চ্যাম্পিয়নের মতোই শুরু"
    test_body = "চ্যাম্পিয়ন্স লিগে চ্যাম্পিয়নের মতোই শুরু করেছে রিয়াল মাদ্রিদ। নতুন মৌসুমে গ্রুপপর্বের প্রথম ম্যাচে সহজ জয় পেয়েছে স্প্যানিশ জায়ান্টরা। ঘরের মাঠে ইতালিয়ান জায়ান্ট এএস রোমাকে ৩-০ গোলে উড়িয়ে দিয়েছে প্রতিযোগিতাটির বর্তমান চ্যাম্পিয়নরা। একটি করে গোল করেছেন গ্যারেথ বেল, ইসকো ও ডিয়াজ। গত মৌসুমে টানা তৃতীয়বারের মতো চ্যাম্পিয়ন্স লিগ শিরোপা ঘরে তোলার পর রিয়াল ছেড়েছেন কোচ জিনেদিন জিদান ও সেরা খেলোয়াড় ক্রিস্টিয়ানো রোনালদো। দুজনের অভাব অবশ্য রোমা ম্যাচে টের পায়নি রিয়াল। বার্নাব্যুতে ‘রোনালদো-জিদান’ জুটি ছাড়াই দুর্দান্ত শুরু করেছে লস ব্লাঙ্কোসরা। ম্যাচে ৬২ শতাংশ সময় নিজেদের পায়ে বল রেখে বল দখলেও এগিয়ে ছিল রিয়াল। রোমার গোলমুখে রিয়ালের শট ছিল ৩০টি, যার ১১টিই আবার লক্ষ্যে। তবে বল দখলের সঙ্গে ধারাবাহিক আক্রমণের পরও গোল পেতে সময় লাগে রিয়ালের। ম্যাচের তৃতীয় মিনিটেই সহজ সুযোগ নষ্ট করেন বেল। আর ইসকোর পা থেকে আসা প্রথম গোলটি ছিল ৪৫ মিনিটে। ইসকোর দিনে গোল পেয়েছেন লা লিগায় একের পর এক ম্যাচে গোল পাওয়া গ্যারেথ বেল। চ্যাম্পিয়ন্স লিগও গোল দিয়ে শুরু করলেন ওয়েলস তারকা। দ্বিতীয়ার্ধের ৫৮ মিনিটে লুকা মদ্রিচের পাস থেকে দারুণ ফিনিশিং দেন বেল। রিয়ালের তৃতীয় গোলটি করেন স্কোরার বেলের বদলি হয়ে নামা মারিয়ানো ডিয়াজ। যোগ করা সময়ে মার্সেলোর পাস থেকে পাওয়া বলে বুলেট–গতির শটে রোমার জালে জড়িয়ে খেলা শেষ করেন তিনি। ‘জি’ গ্রুপে ৩ পয়েন্ট নিয়ে গ্রুপপর্বের খেলায় শীর্ষে থাকল স্প্যানিশ জায়ান্টরা। তালিকায় সবার শেষে রোমা। রিয়ালের জয়ের রাতে জিতেছে রোনালদোর জুভেন্টাসও। স্প্যানিশ ক্লাব ভ্যালেন্সিয়ার বিপক্ষে ২-০ গোলে জিতেছে ইতালিয়ান জায়ান্টরা। তবে দলের জয়ের দিনে লাল কার্ড দেখে মাঠ ছেড়েছেন রিয়ালের সাবেক সুপারস্টার।"

    predict_news(test_cat, test_head, test_body)