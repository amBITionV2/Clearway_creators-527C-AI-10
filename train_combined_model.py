import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, roc_auc_score
import joblib

# ===============================
# 1️⃣ Text cleaning
# ===============================
def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#[^\s]+", "", text)
    text = re.sub(r"[^A-Za-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text

# ===============================
# 2️⃣ Load datasets
# ===============================
print("📂 Loading main dataset...")
df1 = pd.read_csv("train_E6oV3lV.csv")

# Detect text and label columns automatically
text_col, label_col = None, None
for c in df1.columns:
    if c.lower() in ("tweet","text","comment","post"):
        text_col = c
    if c.lower() in ("label","class","target","y"):
        label_col = c

if text_col is None or label_col is None:
    raise SystemExit("Cannot find text/label columns in train_E6oV3lV.csv")

df1.rename(columns={text_col: "text", label_col: "label"}, inplace=True)
df1["text"] = df1["text"].astype(str).apply(clean_text)
df1["label"] = df1["label"].astype(int)

print("📂 Loading Bangalore slang dataset...")
df2 = pd.read_excel("Copy of bangalore_slang_dataset_binary_nolang(1).xlsx")

if not set(["base_word", "variations", "label"]).issubset(df2.columns):
    raise SystemExit("Expected columns: base_word, variations, label")

# Flatten variations into text samples
rows = []
for _, row in df2.iterrows():
    base = str(row["base_word"])
    variations = str(row["variations"]).split(",")
    label = int(row["label"])
    for v in variations:
        rows.append({"text": clean_text(v.strip()), "label": label})
df2_flat = pd.DataFrame(rows)

# Combine both datasets
df = pd.concat([df1[["text", "label"]], df2_flat], ignore_index=True)
print(f"✅ Combined dataset size: {len(df)} samples")
print(df["label"].value_counts())

# ===============================
# 3️⃣ Split data
# ===============================
X = df["text"].values
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===============================
# 4️⃣ TF-IDF + Model
# ===============================
word_vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=25000, stop_words='english')
char_vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), max_features=8000)

features = FeatureUnion([
    ('word', word_vectorizer),
    ('char', char_vectorizer)
], n_jobs=-1)

base_svc = LinearSVC(class_weight='balanced', max_iter=5000)
calibrated = CalibratedClassifierCV(estimator=base_svc, cv=3, method='sigmoid')

pipeline = Pipeline([
    ('features', features),
    ('clf', calibrated)
])

# ===============================
# 5️⃣ Train
# ===============================
print("🚀 Training model on combined dataset...")
pipeline.fit(X_train, y_train)

# ===============================
# 6️⃣ Evaluate
# ===============================
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

print("\nClassification report:\n", classification_report(y_test, y_pred))
try:
    auc = roc_auc_score(y_test, y_prob)
    print("ROC AUC:", auc)
except Exception:
    pass

# ===============================
# 7️⃣ Save model
# ===============================
joblib.dump(pipeline, "hate_combined_pipeline.joblib")
print("\n✅ Model saved as hate_combined_pipeline.joblib")