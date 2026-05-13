import csv
import os
import time
from urllib.parse import urlparse
from playwright.sync_api import sync_playwright


def extract_category(url):
    try:
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        path = parsed_url.path.strip('/')
        parts = path.split('/')
        if "prothomalo.com" not in domain and domain != "":
            return domain.replace("www.", "").split('.')[0].capitalize()
        excluded_paths = ['video', 'photo', 'advertisement', 'archive', 'author']
        if len(parts) > 0:
            category = parts[0].lower()
            if category in excluded_paths and len(parts) > 1:
                return parts[1].capitalize()
            if category not in excluded_paths:
                return category.capitalize()
    except:
        pass
    return "Others"


def get_existing_data(file_path):
    existing_urls = set()
    last_id = 0
    if not os.path.exists(file_path): return existing_urls, last_id
    with open(file_path, mode='r', encoding='utf-8-sig') as f:
        reader = list(csv.DictReader(f))
        for row in reader:
            if 'URL' in row: existing_urls.add(row['URL'])
            if 'Article ID' in row:
                try:
                    last_id = max(last_id, int(row['Article ID']))
                except:
                    pass
    return existing_urls, last_id


def scrape_to_csv():
    # প্রোজেক্ট স্ট্রাকচার অনুযায়ী পাথ সেট করা হয়েছে
    file_path = 'data/raw_news_data.csv'
    if not os.path.exists('data'): os.makedirs('data')

    homepage = "https://www.prothomalo.com/"
    existing_urls, current_id = get_existing_data(file_path)
    print(f"📊 সংগৃহীত: {len(existing_urls)}, সর্বশেষ আইডি: {current_id}")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(homepage, wait_until="networkidle")
        links = page.locator("h3.headline-title a").all()
        current_urls = [(href if href.startswith("http") else homepage + href.strip('/'))
                        for link in links if (href := link.get_attribute("href"))
                        and (href if href.startswith("http") else homepage + href.strip('/')) not in existing_urls]

        print(f"🚀 নতুন {len(current_urls)}টি খবর পাওয়া গেছে। প্রসেসিং শুরু হচ্ছে...")

        for url in current_urls:
            try:
                page.goto(url, wait_until="domcontentloaded")

                title = page.locator("h1").inner_text().strip()
                category = extract_category(url)

                # --- ১. লেখক এবং লোকেশন সংগ্রহ ---
                try:
                    author_element = page.locator(".contributor-name").first
                    author = author_element.inner_text().strip() if author_element.count() > 0 else "N/A"

                    location_element = page.locator(".author-location").first
                    location = location_element.inner_text().strip() if location_element.count() > 0 else "N/A"
                except:
                    author, location = "N/A", "N/A"

                # --- ২. তারিখ এবং সময় সংগ্রহ (Precise Extraction) ---
                time_element = page.locator("time[datetime]").first
                if time_element.count() > 0:
                    # ISO Format: 2026-02-24T09:08:11+06:00
                    raw_datetime = time_element.get_attribute("datetime")
                    # T সরিয়ে স্পেস দেয়া এবং টাইমজোন বাদ দেয়া (Result: 2026-02-24 09:08:11)
                    pub_date = raw_datetime.replace('T', ' ').split('+')[0]
                else:
                    pub_date = "N/A"

                # --- ৩. বডি এবং ইমেজ সংগ্রহ ---
                body = " ".join(page.locator(".story-element.story-element-text p").all_inner_texts())

                image_locs = page.locator("picture.qt-image:not(.default) img").all()
                images = " | ".join(["https:" + i.get_attribute("src") if i.get_attribute("src") and i.get_attribute(
                    "src").startswith("//") else i.get_attribute("src") for i in image_locs if i.get_attribute("src")])

                captions = " | ".join(
                    [i.get_attribute("alt").strip() if i.get_attribute("alt") else "No description" for i in
                     image_locs])

                # --- ৪. ডেটা সেভ করা ---
                current_id += 1
                file_exists = os.path.isfile(file_path)
                with open(file_path, mode='a', newline='', encoding='utf-8-sig') as f:
                    fieldnames = ['Article ID', 'Category', 'Title', 'Author', 'Location', 'Date', 'Body', 'Image URLs',
                                  'Captions', 'URL', 'Label']
                    writer = csv.DictWriter(f, fieldnames=fieldnames)

                    if not file_exists: writer.writeheader()

                    writer.writerow({
                        'Article ID': current_id,
                        'Category': category,
                        'Title': title,
                        'Author': author,
                        'Location': location,
                        'Date': pub_date,
                        'Body': body,
                        'Image URLs': images,
                        'Captions': captions,
                        'URL': url,
                        'Label': '1'
                    })

                print(f"✅ Saved ID {current_id}: {title[:30]}... | Time: {pub_date}")
                time.sleep(1)

            except Exception as e:
                print(f"❌ Error at {url}: {e}")

        browser.close()


if __name__ == "__main__":
    scrape_to_csv()