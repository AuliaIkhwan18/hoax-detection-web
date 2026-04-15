import download_models

from flask import Flask, render_template, request
import torch
import pickle
from transformers import BertTokenizer, BertForSequenceClassification

app = Flask(__name__)

# ==========================
# LOAD SVM MODEL
# ==========================

vectorizer = pickle.load(
    open("tfidf_vectorizer.pkl", "rb")
)

svm_model = pickle.load(
    open("svm_model.pkl", "rb")
)

# ==========================
# LOAD BERT MODEL
# ==========================

device = torch.device("cpu")

tokenizer = BertTokenizer.from_pretrained(
    "bert-base-multilingual-uncased"
)

bert_model = BertForSequenceClassification.from_pretrained(
    "bert-base-multilingual-uncased",
    num_labels=2
)

bert_model.load_state_dict(
    torch.load(
        "bert_hoax_classifier.pth",
        map_location=device
    ),
    strict=False
)

bert_model.to(device)
bert_model.eval()

# ==========================
# PREDICT FUNCTION (SVM)
# ==========================

def predict_svm(text):

    vec = vectorizer.transform([text])

    pred = svm_model.predict(vec)[0]

    return "Hoax" if pred == 0 else "Valid"


# ==========================
# PREDICT FUNCTION (BERT)
# ==========================

def predict_bert(text):

    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():

        outputs = bert_model(**inputs)

        pred = torch.argmax(
            outputs.logits,
            dim=1
        ).item()

    return "Hoax" if pred == 0 else "Valid"


# ==========================
# ROUTE
# ==========================
@app.route("/", methods=["GET", "POST"])
def index():

    prediction = None
    model_choice = "bert"

    if request.method == "POST":

        text = request.form["news"]
        model_choice = request.form["model"]

        print("Model dipilih:", model_choice)

        if model_choice == "svm":

            prediction = predict_svm(text)

        else:

            prediction = predict_bert(text)

    return render_template(
        "index.html",
        prediction=prediction,
        model_choice=model_choice
    )


# ==========================
# RUN APP
# ==========================

if __name__ == "__main__":

    app.run(host="0.0.0.0", port=5000)