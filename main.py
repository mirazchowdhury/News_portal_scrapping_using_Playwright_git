import os
import subprocess
import sys

def run_pipeline():
    # রুট ডিরেক্টরি সেট করা
    base_dir = os.path.dirname(os.path.abspath(__file__))
    python_exe = sys.executable  # বর্তমান ভার্চুয়াল এনভায়রনমেন্টের পাইথন

    print("🚀 Bangla Fake News Detection Pipeline Started...\n")

    # Phase 1: Scraping
    print("--- Phase 1: Scraping Data ---")
    subprocess.run([python_exe, os.path.join(base_dir, "src", "scraper", "scraper.py")])

    # Phase 2: Merging
    print("\n--- Phase 2: Merging Datasets ---")
    subprocess.run([python_exe, os.path.join(base_dir, "src", "processor", "merged_dataset.py")])

    # Phase 3: Training (আপনার এরর এখানে হচ্ছে)
    print("\n--- Phase 3: Training Model (This may take time) ---")
    # os.path.join ব্যবহার করলে স্পেস বা স্ল্যাশ নিয়ে সমস্যা হয় না
    train_script = os.path.join(base_dir, "src", "ml", "model_save.py")
    subprocess.run([python_exe, train_script])

    # Phase 4: Verifying
    print("\n--- Phase 4: Verifying Scraped Data ---")
    subprocess.run([python_exe, os.path.join(base_dir, "src", "ml", "classifier.py")])

    print("\n✅ All processes completed successfully!")

if __name__ == "__main__":
    run_pipeline()