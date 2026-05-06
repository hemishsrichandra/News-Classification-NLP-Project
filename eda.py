import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

def plot_class_distribution(labels, title):
    counter = Counter(labels)
    
    classes = list(counter.keys())
    counts = list(counter.values())

    plt.bar(classes, counts)
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Count")

    plt.savefig("plots/class_distribution.png")
    plt.close()

def plot_text_length_distribution(texts):
    lengths = [len(t.split()) for t in texts]

    plt.hist(lengths, bins=30)
    plt.title("Text Length Distribution")
    plt.xlabel("Number of Words")
    plt.ylabel("Frequency")
    plt.savefig("plots/text_length_distribution.png")
    plt.close()

def show_sample_texts(texts, labels):
    print("\nSample Data:\n")
    for i in range(5):
        print(f"Label: {labels[i]}")
        print(f"Text: {texts[i][:200]}...\n")

from sklearn.manifold import TSNE

def plot_tsne(X, y, title, filename, label_names=None):
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_2d = tsne.fit_transform(X)

    plt.figure(figsize=(8, 6))

    for label in np.unique(y):
        idx = y == label
        name = label_names[label] if label_names else label
        plt.scatter(X_2d[idx, 0], X_2d[idx, 1], label=name, alpha=0.6)

    plt.title(title)
    plt.legend()
    plt.savefig(filename)
    plt.close()
