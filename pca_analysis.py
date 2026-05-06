import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD

def apply_svd(X, n=100):
    svd = TruncatedSVD(n_components=n)
    X_new = svd.fit_transform(X)
    return X_new, svd

def plot_variance(reducer, title, filename):
    plt.plot(reducer.explained_variance_ratio_)
    plt.title(title)
    plt.xlabel("Components")
    plt.ylabel("Variance")
    plt.savefig(filename)
    plt.close()