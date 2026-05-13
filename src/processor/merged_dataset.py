import pandas as pd
import os
import csv
import numpy as np


def merge_datasets():
    # শুরুতে খালি ডাটাফ্রেম নিয়ে রাখা
    my_df_final = pd.DataFrame()
    auth_7k_final = pd.DataFrame()
    fake_df_final = pd.DataFrame()

    # ১. প্রোথম আলোর ডেটা লোড (Current Authentic News)
    prothom_file = 'data/raw_news_data.csv'
    if os.path.exists(prothom_file):
        try:
            df_prothom = pd.read_csv(prothom_file, encoding='utf-8-sig')
            my_df_final = df_prothom[['Title', 'Body', 'Category']].copy()
            my_df_final.columns = ['headline', 'content', 'category']
            my_df_final['label'] = 1
            print(f"✅ প্রোথম আলো থেকে পাওয়া গেছে: {len(my_df_final)} টি রো।")
        except Exception as e:
            print(f"❌ প্রোথম আলো ফাইল লোড এরর: {e}")

    # ২. LabeledAuthentic-7K.csv থেকে ১০০০টি ডেটা যোগ করা (To balance the dataset)
    auth_file = 'data/LabeledAuthentic-7K.csv'
    if os.path.exists(auth_file):
        try:
            df_auth_7k = pd.read_csv(auth_file, encoding='utf-8-sig')
            # ১০০০টি র্যান্ডম স্যাম্পল নেওয়া
            auth_7k_final = df_auth_7k.sample(n=1000, random_state=42).copy()
            # শুধুমাত্র প্রয়োজনীয় কলাম রাখা (যদি কলামের নাম ভিন্ন থাকে তবে রিনেম করে নেওয়া)
            auth_7k_final = auth_7k_final[['headline', 'content', 'category']]
            auth_7k_final['label'] = 1
            print(f"✅ ৭কে অথেন্টিক ফাইল থেকে ১০০০টি র্যান্ডম খবর যোগ করা হয়েছে।")
        except Exception as e:
            print(f"❌ অথেন্টিক ফাইল লোড এরর: {e}")

    # ৩. LabeledFake-1K.csv লোড করা (Fake News)
    fake_file = 'data/LabeledFake-1K.csv'
    if os.path.exists(fake_file):
        try:
            df_fake = pd.read_csv(fake_file, encoding='utf-8-sig')
            fake_df_final = pd.DataFrame()
            fake_df_final['headline'] = df_fake['headline']
            fake_df_final['content'] = df_fake['content']
            fake_df_final['category'] = df_fake['category'] if 'category' in df_fake.columns else "Others"
            fake_df_final['label'] = 0
            print(f"✅ ফেক ডেটাসেট থেকে পাওয়া গেছে: {len(fake_df_final)} টি খবর।")
        except Exception as e:
            print(f"❌ ফেক ফাইল লোড এরর: {e}")

    # ৪. ডেটা মার্জ করা
    all_data_frames = [my_df_final, auth_7k_final, fake_df_final]
    # শুধুমাত্র যে ডাটাফ্রেমগুলো খালি না সেগুলোকে ফিল্টার করা
    valid_dfs = [df for df in all_data_frames if not df.empty]

    if valid_dfs:
        merged_data = pd.concat(valid_dfs, ignore_index=True)

        # ৫. ক্লিনিং
        merged_data['headline'] = merged_data['headline'].replace(['', ' '], np.nan)
        merged_data = merged_data.dropna(subset=['headline'])
        # ডুপ্লিকেট বাদ দেওয়া (৭কে ডেটার সাথে বর্তমান নিউজের ডুপ্লিকেট থাকতে পারে)
        merged_data = merged_data.drop_duplicates(subset=['headline'])

        # ডেটা এলোমেলো করা (সাফলিং)
        merged_data = merged_data.sample(frac=1, random_state=42).reset_index(drop=True)

        # ৬. সেভ করা
        output_path = 'data/prothom_alo_with_fake_data.csv'
        merged_data.to_csv(output_path, index=False, encoding='utf-8-sig', quoting=csv.QUOTE_ALL)

        # পরিসংখ্যান দেখানো
        print("\n" + "=" * 45)
        print("📊 মার্জ করা ডেটাসেটের চূড়ান্ত পরিসংখ্যান:")
        counts = merged_data['label'].value_counts()
        print(f"✅ Total Authentic (1): {counts.get(1, 0)} টি")
        print(f"❌ Total Fake      (0): {counts.get(0, 0)} টি")
        print(f"🚀 মোট ইউনিক নিউজ  : {len(merged_data)} টি")
        print("=" * 45 + "\n")
    else:
        print("❌ মার্জ করার মতো কোনো ডেটা পাওয়া যায়নি!")


if __name__ == "__main__":
    merge_datasets()