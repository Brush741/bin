# bin

### 1. WAP for preprocessing of a text document such as stop word removal, stemming. --------------------------

import re
try:
    from ntlk.stem import PorterStemmer
    ps=PorterStemmer()
    use_ntlk=True
except Exception:
    use_ntlk=False

STOPWORDS={'a','an','the','that','this','is','are','with','to','be','or','on','by','and'}

def tokenize(text):
    return re.findall(r"\b[a-zA-z]+\b",text.lower())

def SimpleStem(word):
    for suf in ('ing','ed','ly','es','s'):
        if word.endswith(suf) and len(word)-len(suf) >= 3:
            return word[:-len(suf)]
    return word
def preprocess(text,remove_stopwords=True,do_stem=True):
    tokens=tokenize(text)
    if remove_stopwords:
        tokens=[t for t in tokens if t not in STOPWORDS]
    if do_stem:
        if use_ntlk:
            tokens=[ps.stem(t) for t in tokens]
        else:
            tokens=[SimpleStem(t) for t in tokens]
    return tokens

if __name__ =="__main__":
    s="This is the simple example: The cats were running quickly"
    print("orignal:",s)
    print("tokens:",preprocess(s))


### 2. WAP for retrieval of documents using inverted files.---------------------------

# Simple Document Retrieval using Inverted Index

def build_index(docs):
    index = {}
    for i, text in docs.items():
        for w in text.lower().split():
            index.setdefault(w, set()).add(i)
    return index

def search(query, index):
    words = query.lower().split()
    result = set.intersection(*(index.get(w, set()) for w in words)) if words else set()
    return result or "No match found"

# Example
docs = {
    1: "the cat sat on the mat",
    2: "dog and cat playing",
    3: "the quick brown fox",
    4: "cat dog"
}

index = build_index(docs)
print("Inverted Index:", index)
q = input("Enter words to search: ")
print("Documents found:", search(q, index))


### 3. WAP to construct a Baysian network considering medical data. Use this model to demonstrate the diagnosis of heart patients using the standard Heart Disease DataSet.--------------------

# heart_naive_bayes.py
# Train Gaussian Naive Bayes on a heart disease CSV.
# CSV expected: features columns and a label column named 'target' (0 = no disease, 1 = disease)
# Example datasets online often follow this format.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def load_data(path=r"C:\Users\aryan\Downloads\archive (6)\heart.csv"):
    df = pd.read_csv(path)
    # If 'target' is not present, adjust accordingly
    if 'target' not in df.columns:
        raise ValueError("CSV must contain 'target' column (0/1).")
    X = df.drop(columns=['target'])
    y = df['target']
    return X, y

def train_and_eval(path=r"C:\Users\aryan\Downloads\archive (6)\heart.csv"):
    X, y = load_data(path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model = GaussianNB()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))
    print("Classification report:\n", classification_report(y_test, preds))
    return model

if __name__ == "__main__":
    # Run with your CSV file in same folder named heart.csv
    model = train_and_eval(r"C:\Users\aryan\Downloads\archive (6)\heart.csv")
    # Example predict (replace values with real feature-vector)
    # sample = [[63,1,3,145,233,1,0,150,0,2.3,0,0,1]]  # customize length/order to your CSV
    # print("Predicted:", model.predict(sample))

### 4. Implement Agglomerative hierarchical clustering algorithm using appropriate dataset.----------------------

# agglomerative_cluster.py
# Uses sklearn's AgglomerativeClustering with a simple dataset (Iris)
# Also shows a dendrogram using scipy (optional).

from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

def cluster_iris(n_clusters=3):
    data = load_iris()
    X = data.data
    Xs = StandardScaler().fit_transform(X)
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels = model.fit_predict(Xs)
    return Xs, labels

def plot_dendrogram(Xs):
    Z = linkage(Xs, method='ward')
    plt.figure(figsize=(8, 4))
    dendrogram(Z, truncate_mode='level', p=5)
    plt.title("Dendrogram (truncated)")
    plt.xlabel("samples")
    plt.ylabel("distance")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    Xs, labels = cluster_iris(3)
    print("Cluster labels (first 20):", labels[:20])
    plot_dendrogram(Xs)

### 5. Implement Page rank Algorithm.-------------------------------------

# pip install beautifulsoup4
from bs4 import BeautifulSoup

# Sample HTML pages (offline)
pages_html = {
    'A': '<a href="B">B</a> <a href="C">C</a>',
    'B': '<a href="C">C</a>',
    'C': '<a href="A">A</a>',
    'D': '<a href="C">C</a>'
}

# Extract links using BeautifulSoup
pages = {}
for name, html in pages_html.items():
    soup = BeautifulSoup(html, 'html.parser')
    pages[name] = [a['href'] for a in soup.find_all('a')]

# --- PageRank ---
d = 0.85
N = len(pages)
ranks = {p: 1/N for p in pages}

for _ in range(30):
    new = {}
    for p in pages:
        incoming = [x for x, links in pages.items() if p in links]
        new[p] = (1-d)/N + d * sum(ranks[i]/len(pages[i]) for i in incoming)
    ranks = new

print("PageRank Results:")
for p, r in ranks.items():
    print(p, ":", round(r, 4))
