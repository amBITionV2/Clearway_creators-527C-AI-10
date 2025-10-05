from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import re

app = Flask(__name__)

# ==========================
# Load trained model
# ==========================
print("📦 Loading trained hate speech model...")
model = joblib.load("hate_combined_pipeline.joblib")
print("✅ Model loaded successfully!")

# ==========================
# Load slang dataset
# ==========================
print("📂 Loading Bangalore slang dataset...")
df_slang = pd.read_excel("Copy of bangalore_slang_dataset_binary_nolang(1).xlsx")

df_slang["base_word"] = df_slang["base_word"].astype(str).str.lower().str.strip()
df_slang["variations"] = df_slang["variations"].astype(str).str.lower().str.strip()

BASE_WORDS = set()
for _, row in df_slang.iterrows():
    BASE_WORDS.add(row["base_word"])
    for v in str(row["variations"]).split(","):
        BASE_WORDS.add(v.strip())

print(f"✅ Loaded {len(BASE_WORDS)} slang/variation words.")


# ==========================
# Utilities
# ==========================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def contains_base_word(text):
    text = clean_text(text)
    for base in BASE_WORDS:
        if base in text.split() or base in text:
            return True
    return False


# ==========================
# Routes
# ==========================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/check_comment", methods=["POST"])
def check_comment():
    data = request.get_json()
    comment = data.get("comment", "")

    if not comment.strip():
        return jsonify({"status": "error", "message": "⚠ Empty comment!"}), 400

    if contains_base_word(comment):
        return jsonify({
            "status": "blocked",
            "method": "base_word_match",
            "probability": 1.0,
            "message": "⚠ Contains offensive Bangalore slang word."
        })

    pred_prob = model.predict_proba([comment])[0][1]
    pred_label = int(pred_prob >= 0.5)

    if pred_label == 1:
        return jsonify({
            "status": "blocked",
            "method": "trained_model",
            "probability": round(float(pred_prob), 3),
            "message": "⚠ Hate or offensive speech detected."
        })
    else:
        return jsonify({
            "status": "allowed",
            "method": "trained_model",
            "probability": round(float(pred_prob), 3),
            "message": "✅ Comment allowed."
        })


if __name__ == "__main__":
    app.run(debug=True)