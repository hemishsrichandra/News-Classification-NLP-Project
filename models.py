from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

def get_models():
    return {
        "logreg": LogisticRegression(max_iter=1000, random_state=42),
        "svm_linear": SVC(kernel="linear", random_state=42),
        "svm_rbf": SVC(kernel="rbf", random_state=42),
        "knn": KNeighborsClassifier(n_neighbors=5),
        "kmeans": KMeans(n_clusters=4, random_state=42)
    }