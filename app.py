from flask import Flask, render_template, request
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Initialize Flask app
app = Flask(__name__)

# Load tokenizer and label map
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("label_map.pkl", "rb") as f:
    label_map = pickle.load(f)

# Inverse label map for decoding predictions
inv_label_map = {v: k for k, v in label_map.items()}

# Load a single trained model (you can add more if needed)
model_path = "C:/Users/TAMMINENI SIVA KUMAR/Downloads/Micro IT Internship/Sentiment Analysis/bilstm_fold_4.keras"
model = load_model(model_path)

# Max sequence length (should match the training)
MAX_LEN = 100

@app.route("/", methods=["GET", "POST"])
def predict():
    comment = ""
    sentiment = ""
    probability = ""

    if request.method == "POST":
        comment = request.form.get("comment", "").strip()
        if comment:
            # Preprocess the comment
            seq = tokenizer.texts_to_sequences([comment])
            padded_seq = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')

            # Predict sentiment
            preds = model.predict(padded_seq)[0]
            pred_idx = np.argmax(preds)
            sentiment = inv_label_map[pred_idx]
            probability = f"{preds[pred_idx]:.4f}"

    return render_template("index.html", comment=comment, sentiment=sentiment, probability=probability)

if __name__ == "__main__":
    app.run(debug=True)
