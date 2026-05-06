import numpy as np
import os
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from collections import Counter

from models import get_models
from pca_analysis import apply_svd, plot_variance
from eda import plot_class_distribution, plot_text_length_distribution, show_sample_texts, plot_tsne

# ======================
# KMeans Helper Function 
# ======================
def train_and_map_kmeans(model, X_train, y_train, X_test):
    # Ensure y_train is a numpy array to avoid HuggingFace list indexing errors
    y_train = np.array(y_train) 
    model.fit(X_train)
    train_preds = model.predict(X_train)
    
    label_map = {}
    for cluster in np.unique(train_preds):
        indices = np.where(train_preds == cluster)[0]
        if len(indices) > 0:
            true_labels = y_train[indices]
            most_common = Counter(true_labels).most_common(1)[0][0]
            label_map[cluster] = most_common
        else:
            label_map[cluster] = 0
            
    test_preds = model.predict(X_test)
    return np.array([label_map.get(c, 0) for c in test_preds])

def evaluate_models(X_train, y_train, X_test, y_test, name_prefix):
    results = {}
    for name, model in get_models().items():
        print(f"\n{name} ({name_prefix})")
        if name == "kmeans":
            preds = train_and_map_kmeans(model, X_train, y_train, X_test)
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
        
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")
        f1_w = f1_score(y_test, preds, average="weighted")
        print(f"Macro F1: {f1:.4f} | Weighted F1: {f1_w:.4f} | Accuracy: {acc:.4f}")
        results[name] = {"acc": acc, "f1": f1}
    return results

# ======================
# 1. LOAD DATA
# ======================
os.makedirs("plots", exist_ok=True)
# AG News labels correspond to World, Sports, Business, Sci/Tech
label_names = ["World", "Sports", "Business", "Sci/Tech"]

dataset = load_dataset("ag_news")
train = dataset["train"].shuffle(seed=42).select(range(8000))
test = dataset["test"].shuffle(seed=42).select(range(2000))

X_train_text, y_train = train["text"], np.array(train["label"])
X_test_text, y_test = test["text"], np.array(test["label"])

y_sample = y_train[:2000] # Subsample labels for t-SNE

# Dictionary to hold all final metrics
all_results = {}

# ======================
# 2. EDA
# ======================
plot_class_distribution(y_train, "Train Class Distribution")
plot_text_length_distribution(X_train_text)
show_sample_texts(X_train_text, y_train)

# ======================
# 3. BAG OF WORDS
# ======================
print("\n===== 1. BAG OF WORDS =====")
bow = CountVectorizer(max_features=5000, stop_words='english')
X_train_bow = bow.fit_transform(X_train_text)
X_test_bow = bow.transform(X_test_text)
all_results["BoW"] = evaluate_models(X_train_bow, y_train, X_test_bow, y_test, "BoW")

# ======================
# 4. TF-IDF & SVD
# ======================
print("\n===== 2a. TF-IDF (BEFORE SVD) =====")
tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=8000, stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train_text)
X_test_tfidf = tfidf.transform(X_test_text)

# T-SNE Plot 1
plot_tsne(X_train_tfidf[:2000].toarray(), y_sample, "t-SNE (TF-IDF Raw)", "plots/tsne_tfidf.png", label_names)

all_results["TF-IDF"] = evaluate_models(X_train_tfidf, y_train, X_test_tfidf, y_test, "TF-IDF Raw")

print("\n===== 2b. TF-IDF (AFTER SVD) =====")
X_train_svd, svd = apply_svd(X_train_tfidf, 100)
X_test_svd = svd.transform(X_test_tfidf)

plot_variance(svd, "Explained Variance (SVD)", "plots/svd_variance.png")

# T-SNE Plot 2
plot_tsne(X_train_svd[:2000], y_sample, "t-SNE (TF-IDF + SVD)", "plots/tsne_svd.png", label_names)

all_results["TF-IDF+SVD"] = evaluate_models(X_train_svd, y_train, X_test_svd, y_test, "TF-IDF+SVD")

# Confusion Matrix for Best Sparse Model
print("\nGenerating Confusion Matrix (TF-IDF + SVD - Best Model)...")
best_svm = get_models()["svm_rbf"]
best_svm.fit(X_train_svd, y_train)
cm = confusion_matrix(y_test, best_svm.predict(X_test_svd))
disp = ConfusionMatrixDisplay(cm, display_labels=label_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix (TF-IDF + SVD)")
plt.savefig("plots/confusion_matrix_tfidf.png")
plt.close()

# ======================
# 5. BERT & PCA
# ======================
try:
    X_train_bert = np.load("embeddings/train.npy")
    X_test_bert = np.load("embeddings/test.npy")
except FileNotFoundError:
    print("\n[!] Embeddings not found. Please run extract_embeddings.py first.")
    exit()

print("\n===== 3a. BERT CLS (BEFORE PCA) =====")
# T-SNE Plot 3
plot_tsne(X_train_bert[:2000], y_sample, "t-SNE (BERT Raw)", "plots/tsne_bert_raw.png", label_names)

all_results["BERT"] = evaluate_models(X_train_bert, y_train, X_test_bert, y_test, "BERT Raw")

print("\n===== 3b. BERT CLS (AFTER PCA) =====")
pca = PCA(n_components=100)
X_train_bert_pca = pca.fit_transform(X_train_bert)
X_test_bert_pca = pca.transform(X_test_bert)

plot_variance(pca, "PCA Explained Variance (BERT)", "plots/pca_variance_bert.png")

# T-SNE Plot 4
plot_tsne(X_train_bert_pca[:2000], y_sample, "t-SNE (BERT + PCA)", "plots/tsne_bert_pca.png", label_names)

all_results["BERT+PCA"] = evaluate_models(X_train_bert_pca, y_train, X_test_bert_pca, y_test, "BERT+PCA")

# ======================
# 6. FINAL COMPARISON & EXPORT
# ======================
models_list = list(get_models().keys())
experiments = list(all_results.keys()) # BoW, TF-IDF, TF-IDF+SVD, BERT, BERT+PCA
x = np.arange(len(models_list))
width = 0.15 

# Plot Accuracy
plt.figure(figsize=(14, 7))
for i, exp in enumerate(experiments):
    accs = [all_results[exp][m]["acc"] for m in models_list]
    plt.bar(x + (i - 2) * width, accs, width, label=exp)
plt.xticks(x, models_list)
plt.title("Accuracy Comparison Across All Feature Spaces")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("plots/accuracy_comparison.png")
plt.close()

# Plot F1
plt.figure(figsize=(14, 7))
for i, exp in enumerate(experiments):
    f1s = [all_results[exp][m]["f1"] for m in models_list]
    plt.bar(x + (i - 2) * width, f1s, width, label=exp)
plt.xticks(x, models_list)
plt.title("Macro F1 Comparison Across All Feature Spaces")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("plots/f1_comparison.png")
plt.close()

# Save Comprehensive CSV
csv_data = {"Model": models_list}
for exp in experiments:
    csv_data[f"{exp} Accuracy"] = [all_results[exp][m]["acc"] for m in models_list]
    csv_data[f"{exp} Macro F1"] = [all_results[exp][m]["f1"] for m in models_list]
    
df = pd.DataFrame(csv_data)
df.to_csv("results.csv", index=False)
print("\nFinal results (Accuracy and F1) saved to results.csv")

# ======================
# 7. INFERENCE
# ======================
def predict_text(text, model, vectorizer, reducer=None):
    X = vectorizer.transform([text])
    if reducer: X = reducer.transform(X)
    pred = model.predict(X)[0]
    return {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}[pred]

while(True):
    print('\nIf you want to exit, enter "exit" or "quit"')
    to_predict = input("Enter the news snippet you want to classify: ")

    if(to_predict == "exit" or to_predict == "quit"):
        break

    print("\nPrediction:")
    print(predict_text(to_predict, best_svm, tfidf, svd))