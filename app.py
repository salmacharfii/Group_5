import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Review Sentiment", page_icon="💬", layout="wide")

COLORS = {"negative": "#e34948", "neutral": "#9a9893", "positive": "#2a78d6"}
ORDER = ["negative", "neutral", "positive"]
EXAMPLES = {
    "😊 Positive": "Great sound quality and the battery lasts all day. Totally worth the price!",
    "😐 Neutral": "It works as described. Nothing special, but it does the job.",
    "😠 Negative": "Stopped charging after two weeks. Customer service never answered.",
}


@st.cache_resource
def load_model():
    return joblib.load("Models/sentiment_model.joblib")


model = load_model()
vectorizer, clf = model[:-1], model[-1]          # TF-IDF part and Logistic Regression part
feature_names = vectorizer.get_feature_names_out()


def donut(labels, values, center_text):
    pairs = sorted(zip(labels, values), key=lambda x: ORDER.index(str(x[0]).lower()))
    labels = [str(l).capitalize() for l, _ in pairs]
    values = [v for _, v in pairs]
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.6, sort=False, direction="clockwise",
        marker=dict(colors=[COLORS[l.lower()] for l in labels], line=dict(color="white", width=2)),
        textinfo="label+percent", textposition="outside",
        hovertemplate="%{label}: %{value:,} (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        showlegend=True, legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center"),
        annotations=[dict(text=center_text, x=0.5, y=0.5, showarrow=False, font=dict(size=20))],
        margin=dict(t=30, b=30, l=30, r=30), height=360,
    )
    return fig


def word_contributions(text, class_name, top_n=8):
    """Which words in this review pushed the model toward class_name."""
    x = vectorizer.transform([text])
    class_idx = list(clf.classes_).index(class_name)
    contrib = x.multiply(clf.coef_[class_idx]).tocsr()
    idx, vals = contrib.indices, contrib.data
    order = np.argsort(-np.abs(vals))[:top_n]
    return pd.DataFrame({"word": feature_names[idx[order]], "weight": vals[order]}).sort_values("weight")


def contributions_chart(df, class_name):
    color = COLORS[class_name.lower()]
    fig = go.Figure(go.Bar(
        x=df["weight"], y=df["word"], orientation="h",
        marker=dict(color=[color if w > 0 else "#c3c2b7" for w in df["weight"]], cornerradius=4),
        hovertemplate="%{y}: %{x:+.2f}<extra></extra>",
    ))
    fig.update_layout(
        height=300, margin=dict(t=10, b=30, l=10, r=10),
        xaxis=dict(title=f"← against {class_name}   |   towards {class_name} →", zeroline=True, zerolinecolor="#999", showgrid=False),
        yaxis=dict(showgrid=False),
    )
    return fig

# Thresholds for the verdict (tune these to your data)
RECOMMEND_MIN_POS = 0.70      # at least 70% positive ...
RECOMMEND_MAX_NEG = 0.15      # ... and at most 15% negative
REJECT_MIN_NEG = 0.30         # 30%+ negative -> not recommended


def key_terms(texts, class_name, top_n=4):
    """Words that most drive reviews of this class (e.g. what people complain about)."""
    if len(texts) == 0:
        return []
    X = vectorizer.transform(texts)
    class_idx = list(clf.classes_).index(class_name)
    scores = np.asarray(X.sum(axis=0)).ravel() * clf.coef_[class_idx]
    top = np.argsort(-scores)[:top_n]
    return [feature_names[i] for i in top if scores[i] > 0]


def product_verdict(group):
    n = len(group)
    share = group["sentiment"].value_counts(normalize=True)
    pos, neu, neg = (float(share.get(c, 0)) for c in ORDER[::-1])
    praise = key_terms(group.loc[group["sentiment"] == "positive", "_text"], "positive")
    complaints = key_terms(group.loc[group["sentiment"] == "negative", "_text"], "negative")

    praise_txt = f" Customers especially mention **{', '.join(praise)}**." if praise else ""
    complaint_txt = f" Common complaints: **{', '.join(complaints)}**." if complaints else ""

    if pos >= RECOMMEND_MIN_POS and neg <= RECOMMEND_MAX_NEG:
        return "recommend", (f"✅ **Recommended.** {pos:.0%} of {n:,} reviews are positive "
                             f"and only {neg:.0%} negative.{praise_txt}")
    if neg >= REJECT_MIN_NEG:
        return "reject", (f"❌ **Not recommended.** {neg:.0%} of {n:,} reviews are negative."
                          f"{complaint_txt}")
    return "mixed", (f"⚠️ **Mixed reviews.** {pos:.0%} positive, {neu:.0%} neutral, {neg:.0%} negative "
                     f"across {n:,} reviews.{praise_txt}{complaint_txt}")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("About the model")
    st.write("**Type:** TF-IDF + Logistic Regression")
    st.write("**Training data:** 27,700 Amazon reviews")
    st.write("**Classes:** negative · neutral · positive")
    st.caption("Group 5 · Ironhack AI Engineering")

st.title("💬 Customer Review Sentiment")
tab1, tab2, tab3 = st.tabs(["Single review", "Batch analysis", "Model performance"])

# ---------------- Tab 1: single review ----------------
with tab1:
    if "review" not in st.session_state:
        st.session_state.review = EXAMPLES["😠 Negative"]

    st.write("Try an example:")
    cols = st.columns(len(EXAMPLES))
    for col, (name, text) in zip(cols, EXAMPLES.items()):
        if col.button(name, use_container_width=True):
            st.session_state.review = text

    review = st.text_area("Or paste your own product review", key="review", height=120)

    if st.button("Predict", type="primary") and review.strip():
        probs = model.predict_proba([review])[0]
        pred = model.classes_[probs.argmax()]
        conf = probs.max()

        left, right = st.columns(2)
        with left:
            st.subheader("Prediction")
            st.plotly_chart(donut(model.classes_, probs, f"<b>{str(pred).capitalize()}</b><br>{conf:.0%}"),
                            use_container_width=True)
            if conf < 0.5:
                st.warning("Low confidence: the model is unsure about this review.")
        with right:
            st.subheader("Why this prediction?")
            contrib = word_contributions(review, pred)
            if contrib.empty:
                st.info("None of these words were seen during training.")
            else:
                st.plotly_chart(contributions_chart(contrib, str(pred)), use_container_width=True)
                st.caption("Words with the biggest influence on the prediction.")

# ---------------- Tab 2: batch ----------------
with tab2:
    st.write("Upload a CSV file with one review per row.")
    file = st.file_uploader("CSV file", type="csv")
    if file:
        data = pd.read_csv(file)
        c1, c2 = st.columns(2)
        text_col = c1.selectbox("Review text column", data.columns)
        product_col = c2.selectbox("Product column (optional)",
                                   ["(none - all reviews are one product)"] + list(data.columns))

        if st.button("Analyze all reviews", type="primary"):
            data["_text"] = data[text_col].fillna("").astype(str)
            probs = model.predict_proba(data["_text"])
            data["sentiment"] = model.classes_[probs.argmax(axis=1)]
            data["confidence"] = probs.max(axis=1).round(2)

            # --- Overall summary ---
            counts = data["sentiment"].value_counts()
            m = st.columns(4)
            m[0].metric("Reviews", f"{len(data):,}")
            for col, label in zip(m[1:], ORDER):
                n = int(counts.get(label, 0))
                col.metric(label.capitalize(), f"{n:,}", f"{n / len(data):.0%}", delta_color="off")

            left, right = st.columns([1, 1.4])
            left.plotly_chart(donut(counts.index, counts.values, f"<b>{len(data):,}</b><br>reviews"),
                              use_container_width=True)
            right.dataframe(data[[text_col, "sentiment", "confidence"]], height=360, use_container_width=True)

            # --- Recommendation per product ---
            st.subheader("Recommendation")
            if product_col.startswith("(none"):
                groups = [("All uploaded reviews", data)]
            else:
                top_products = data[product_col].value_counts().head(20).index   # 20 most-reviewed
                groups = [(p, data[data[product_col] == p]) for p in top_products]

            verdicts = []
            for name, group in groups:
                kind, text = product_verdict(group)
                verdicts.append({"product": name, "verdict": kind, "reviews": len(group)})
                with st.container(border=True):
                    st.markdown(f"#### {name}")
                    box = {"recommend": st.success, "reject": st.error, "mixed": st.warning}[kind]
                    box(text)

            st.download_button("⬇ Download results (CSV)",
                               data.drop(columns="_text").to_csv(index=False).encode("utf-8"),
                               "sentiment_results.csv", "text/csv")

# ---------------- Tab 3: performance ----------------
with tab3:
    st.write("Results on the held-out test set (6,925 reviews).")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", "92.4%")
    m2.metric("Macro F1", "0.64")
    m3.metric("Weighted F1", "0.93")
    m4.metric("Test reviews", "6,925")

    st.info("The test set is ~93% positive, so accuracy looks high. "
            "Macro F1 treats all three classes equally and is the fairer score. "
            "Neutral is the hardest class: it is often confused with positive.")

    cm = np.array([[99, 29, 34],
                   [40, 131, 129],
                   [41, 255, 6167]])
    cm_pct = cm / cm.sum(axis=1, keepdims=True)          # % of each true class
    text = [[f"{p:.0%}<br>({n:,})" for p, n in zip(pr, nr)] for pr, nr in zip(cm_pct, cm)]

    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Confusion matrix")
        fig = go.Figure(go.Heatmap(
            z=cm_pct, x=[f"Pred {l}" for l in ORDER], y=[f"True {l}" for l in ORDER],
            colorscale=[[0, "#f0efec"], [1, "#2a78d6"]], zmin=0, zmax=1,
            text=text, texttemplate="%{text}", showscale=False, xgap=2, ygap=2,
            hovertemplate="%{y} → %{x}: %{z:.0%}<extra></extra>",
        ))
        fig.update_layout(height=380, margin=dict(t=10, b=10, l=10, r=10), yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Each row shows where the reviews of that true class ended up (row = 100%).")

    with right:
        st.subheader("Per-class scores")
        scores = pd.DataFrame({
            "Class": ["Negative", "Neutral", "Positive"],
            "Precision": [0.55, 0.32, 0.97],
            "Recall": [0.61, 0.44, 0.95],
            "F1": [0.58, 0.37, 0.96],
            "Support": [162, 300, 6463],
        })
        st.dataframe(scores, hide_index=True, use_container_width=True)