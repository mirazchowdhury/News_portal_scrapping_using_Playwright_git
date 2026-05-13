import streamlit as st
import pandas as pd
import os
import json
import plotly.express as px
import plotly.graph_objects as go


# ১. ফাংশন ডিফিনিশন
def load_data():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_PATH = os.path.join(BASE_DIR, "data", "prothom_alo_verified_master_data.csv")
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return None


def load_metrics():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    METRICS_PATH = os.path.join(BASE_DIR, "models", "best_fake_news_model", "metrics.json")
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            return json.load(f)
    return None


# ২. পেজ সেটআপ এবং ডিজাইন
st.set_page_config(page_title="Bangla Fake News AI", page_icon="🛡️", layout="wide")

# কাস্টম স্টাইল যোগ করা (Colorful Cards & Background)
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { 
        background-color: #ffffff; 
        padding: 20px; 
        border-radius: 15px; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 5px solid #4B90FF;
    }
    div[data-testid="stExpander"] { border-radius: 15px; background-color: #ffffff; }
    </style>
    """, unsafe_allow_html=True)

# ৩. সাইডবার (Filters)
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2592/2592225.png", width=100)
st.sidebar.title("🔍 Control Panel")
st.sidebar.markdown("আপনার পছন্দমতো ডেটা ফিল্টার করুন")

df = load_data()
metrics = load_metrics()

if df is not None:
    # ফিল্টার অপশন
    selected_category = st.sidebar.multiselect("বিভাগ নির্বাচন করুন", options=df['Category'].unique(),
                                               default=df['Category'].unique())
    selected_status = st.sidebar.radio("ভেরিফিকেশন স্ট্যাটাস", ["All", "Real", "Fake"])

    # ফিল্টারিং লজিক
    filtered_df = df[df['Category'].isin(selected_category)]
    if selected_status != "All":
        filtered_df = filtered_df[filtered_df['Verification'] == selected_status]

    # ৪. মেইন কন্টেন্ট
    st.title("🛡️ Bangla Fake News Intelligence Dashboard")
    st.markdown(f"**সর্বশেষ আপডেট:** `{df['Date'].iloc[0] if 'Date' in df.columns else 'N/A'}`")

    # ৫. টপ মেট্রিক্স (Colorful)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("মোট খবর", f"{len(filtered_df)} টি", delta="Scraped", delta_color="normal")
    with col2:
        r_count = len(filtered_df[filtered_df['Verification'] == 'Real'])
        st.metric("✅ অথেন্টিক", r_count, delta="Real News", delta_color="inverse")
    with col3:
        f_count = len(filtered_df[filtered_df['Verification'] == 'Fake'])
        st.metric("❌ ফেক নিউজ", f_count, delta="- Warning", delta_color="normal")
    with col4:
        acc = f"{metrics['accuracy'] * 100:.1f}%" if metrics else "N/A"
        st.metric("🤖 AI Accuracy", acc, help="মডেলের নির্ভুলতার হার")

    st.divider()

    # ৬. মডেল রিপোর্ট (Expander এর ভেতরে সুন্দর করে রাখা)
    if metrics:
        with st.expander("📊 AI Model Performance Details", expanded=False):
            m1, m2, m3, m4 = st.columns(4)
            m1.progress(metrics['accuracy'], text=f"Accuracy: {metrics['accuracy'] * 100:.1f}%")
            m2.progress(metrics['f1_weighted'], text=f"F1-Score: {metrics['f1_weighted'] * 100:.1f}%")
            m3.progress(metrics['precision'], text=f"Precision: {metrics['precision'] * 100:.1f}%")
            m4.progress(metrics['recall'], text=f"Recall: {metrics['recall'] * 100:.1f}%")
            st.info(f"Base Engine: {metrics['model_name']} | Hardware Acceleration: {metrics['device'].upper()}")

    # ৭. ভিজুয়ালাইজেশন (Interactive Charts)
    c1, c2 = st.columns([1, 1])

    with c1:
        st.subheader("🎯 ভেরিফিকেশন রেশিও")
        fig_pie = px.pie(filtered_df, names='Verification', hole=0.5,
                         color='Verification',
                         color_discrete_map={'Real': '#2ecc71', 'Fake': '#e74c3c'},
                         template="plotly_white")
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        st.subheader("📈 ক্যাটাগরি ভিত্তিক সংবাদ")

        # ১. ডেটা প্রিপেয়ার করা (Pandas এর নতুন ভার্সন অনুযায়ী)
        cat_data = filtered_df['Category'].value_counts().reset_index()

        # ২. কলামের নামগুলো নিশ্চিত করা
        # নতুন পান্ডাসে এটি ['Category', 'count'] হয়, আগের ভার্সনে ['index', 'Category'] ছিল
        cat_data.columns = ['CategoryName', 'NewsCount']

        # ৩. গ্রাফ তৈরি করা
        fig_bar = px.bar(cat_data,
                         x='CategoryName', y='NewsCount', color='CategoryName',
                         labels={'NewsCount': 'নিউজের সংখ্যা', 'CategoryName': 'বিভাগ'},
                         color_discrete_sequence=px.colors.qualitative.Pastel)

        st.plotly_chart(fig_bar, use_container_width=True)

    # ৮. ইন্টারঅ্যাক্টিভ টেবিল
    st.subheader("📰 সংবাদের বিস্তারিত তালিকা")
    search_query = st.text_input("শিরোনাম দিয়ে সার্চ করুন...", placeholder="উদা: জনপ্রশাসন")

    if search_query:
        display_df = filtered_df[filtered_df['Title'].str.contains(search_query, case=False, na=False)]
    else:
        display_df = filtered_df

    # স্টাইলিশ ডাটাফ্রেম
    st.dataframe(
        display_df[['Verification', 'Category', 'Title', 'Author', 'URL']],
        column_config={
            "URL": st.column_config.LinkColumn("লিঙ্ক", display_text="সংবাদটি পড়ুন"),
            "Verification": st.column_config.SelectboxColumn("Status", options=["Real", "Fake"])
        },
        use_container_width=True,
        hide_index=True
    )

else:
    st.warning("⚠️ ডেটা পাওয়া যায়নি! অনুগ্রহ করে আগে `main.py` রান করে মডেল ট্রেনিং এবং খবর ভেরিফাই করুন।")