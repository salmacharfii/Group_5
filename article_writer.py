import os
import re
import pandas as pd
from openai import OpenAI

DATA_PATH = os.path.join(os.path.dirname(__file__), "CSVs", "reviews_slim.csv.gz")
df = pd.read_csv(DATA_PATH)

MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"
MIN_REVIEWS = 10          # 👉 use the same value as in your notebook


# ------------------------------------------------------------------
# 👉 PASTE YOUR NOTEBOOK FUNCTIONS HERE, unchanged:
#    product_table, pick_products, facts_text, prompt_v2, check_article
# ------------------------------------------------------------------


# ---------------- NVIDIA API ----------------
def _api_key():
    try:
        import streamlit as st
        return st.secrets["NVIDIA_API_KEY"]
    except Exception:
        return os.environ.get("NVIDIA_API_KEY")


client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=_api_key())


def clean(text):
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()


def write_article(prompt):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": "You are a writer for a product review blog."},
                  {"role": "user", "content": prompt}],
        max_tokens=800,
        temperature=0.2,
        top_p=0.9,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return clean(response.choices[0].message.content)


def categories():
    return sorted(df["category"].dropna().unique())