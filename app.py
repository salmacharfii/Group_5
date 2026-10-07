import joblib
import streamlit as st
import plotly.graph_objects as go

st.title("Customer Review Sentiment")
st.write("Group 5 · Ironhack AI Engineering · TF-IDF + Logistic Regression trained on 27,700 Amazon reviews")

# Fixed colours per sentiment: red = negative, gray = neutral, blue = positive
COLORS = {"negative": "#e34948", "neutral": "#9a9893", "positive": "#2a78d6"}
ORDER = ["negative", "neutral", "positive"]


@st.cache_resource
def load_model():
    return joblib.load("Models/sentiment_model.joblib")


def donut(labels, values, center_text):
    # Always draw in the same order so colours never move around
    pairs = sorted(zip(labels, values), key=lambda x: ORDER.index(str(x[0]).lower()))
    labels = [str(l).capitalize() for l, _ in pairs]
    values = [v for _, v in pairs]
    colors = [COLORS[l.lower()] for l in labels]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.6,                                        # makes it a donut
        sort=False,
        direction="clockwise",
        marker=dict(colors=colors, line=dict(color="white", width=2)),  # small gap between slices
        textinfo="label+percent",                        # labels on the chart, not colour alone
        textposition="outside",
        hovertemplate="%{label}: %{percent}<extra></extra>",
    ))
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center"),
        annotations=[dict(text=center_text, x=0.5, y=0.5, showarrow=False, font=dict(size=22))],
        margin=dict(t=30, b=30, l=30, r=30),
        height=380,
    )
    return fig


model = load_model()

review = st.text_area("Paste a product review", "Stopped charging after two weeks.")

if st.button("Predict"):
    probabilities = model.predict_proba([review])[0]
    prediction = model.classes_[probabilities.argmax()]

    st.subheader(f"Sentiment: {prediction}")
    st.plotly_chart(
        donut(model.classes_, probabilities, f"<b>{str(prediction).capitalize()}</b><br>{probabilities.max():.0%}"),
        use_container_width=True,
    )