import joblib
import gradio as gr

model = joblib.load("sentiment_model.joblib")

def predict(review):
    probabilities = model.predict_proba([review])[0]
    return {label: float(p) for label, p in zip(model.classes_, probabilities)}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=4, label="Paste a product review"),
    outputs=gr.Label(label="Sentiment"),
    title="Customer Review Sentiment",
    examples=[["I love this tablet, the screen is sharp."], ["Stopped charging after two weeks."]],
)

demo.launch()
