import pandas as pd

# ফাইলটি লোড করে দেখা
df = pd.read_csv('prothom_alo_with_fake_data.csv', encoding='utf-8-sig')
print("আসল ডেটা সংখ্যা:", len(df))
print(df['label'].value_counts())