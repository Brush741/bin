ML1 LDA and PCA
LDA:-
# LDA on Iris dataset (short and simple)
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score

# 1) Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# 2) Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# 3) Apply LDA
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

# 4) Predict and check accuracy
y_pred = lda.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# 5) Example prediction
sample = [[5.1, 3.5, 1.4, 0.2]]  # sample input (sepal & petal sizes)
print("Predicted species:", iris.target_names[lda.predict(sample)[0]])

PCA:-
# PCA on Wine Dataset - single cell version
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load dataset
data = pd.read_csv("https://media.geeksforgeeks.org/wp-content/uploads/Wine.csv")

# Split features and target
X = data.drop("Customer_Segment", axis=1)
y = data["Customer_Segment"]

# Standardize and apply PCA
X_scaled = StandardScaler().fit_transform(X)
X_pca = PCA(n_components=2).fit_transform(X_scaled)

# Plot results
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap="viridis", edgecolor="k")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA on Wine Dataset")
plt.show()
---------------------------------------------------x---------------------------------------------x--------------------------------------------------------x----------
ML2  Uber price prediction (Linear, Ridge, Lasso Regression)


# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv(r"C:\Users\aryan\Downloads\archive (10)\uber.csv")

# 1. Pre-processing
df = df.dropna()
df = df[(df['fare_amount'] > 0) & (df['passenger_count'] > 0)]
df = df[(df['pickup_longitude'] != 0) & (df['pickup_latitude'] != 0)]

# Calculate trip distance (approx.)
df['distance'] = np.sqrt((df['dropoff_longitude'] - df['pickup_longitude'])**2 +
                         (df['dropoff_latitude'] - df['pickup_latitude'])**2)

X = df[['distance', 'passenger_count']]
y = df['fare_amount']

# 2. Identify Outliers (Simple visualization)
sns.boxplot(x=df['fare_amount'])
plt.show()

# 3. Check Correlation
print(df[['distance', 'passenger_count', 'fare_amount']].corr())

# 4. Split and Train Models
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression()
ridge = Ridge(alpha=1)
lasso = Lasso(alpha=0.1)

lr.fit(X_train, y_train)
ridge.fit(X_train, y_train)
lasso.fit(X_train, y_train)

# 5. Evaluate Models
models = {'Linear': lr, 'Ridge': ridge, 'Lasso': lasso}

for name, model in models.items():
    y_pred = model.predict(X_test)
    print(f"\n{name} Regression:")
    print("R² Score:", r2_score(y_test, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
------------------------------------------x--------------------------------------------------x------------------------------------------------------------x---------
ML3 → SVM on handwritten digits


# Small & easy SVM on handwritten digits (0–9)

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1) Load data
X, y = load_digits(return_X_y=True)

# 2) Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3) Scale features (helps SVM)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# 4) Train SVM (RBF kernel works well)
clf = SVC(kernel="rbf", gamma="scale", C=10, random_state=42)
clf.fit(X_train, y_train)

# 5) Evaluate
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred))
print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))
---------------------------------------x------------------------------------------------------------x----------------------------------------------------x-----------
ML4 → K-Means 
# K-Means on Iris.csv with elbow method

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 1) Load CSV (from Kaggle Iris dataset)
df = pd.read_csv(r"C:\Users\aryan\OneDrive\Desktop\datasets\Iris.csv")  # columns: Id, SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm, Species
X = df.drop(columns=["Id", "Species"])  # features only

# 2) Elbow method to choose k
inertias = []
K_range = range(1, 11)
for k in K_range:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(X)
    inertias.append(km.inertia_)

plt.plot(list(K_range), inertias, marker="o")
plt.xlabel("k (number of clusters)")
plt.ylabel("Inertia (within-cluster SSE)")
plt.title("Elbow Method for Iris")
plt.show()

# 3) Fit final model (Iris usually has 3 clusters)
k = 3
kmeans = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
labels = kmeans.labels_
print("Cluster centers:\n", kmeans.cluster_centers_)

# 4) (Optional) Compare clusters with true species
print("\nCluster vs Species (rough match, order may differ):")
print(pd.crosstab(labels, df["Species"]))
-----------------------------------------------------x--------------------------------------------------------------x-----------------------------------------------

ML5 → Random Forest (car dataset)


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Read your file (no header). It may be comma or whitespace separated.
df = pd.read_csv(
    r"C:\Users\aryan\Downloads\archive (11)\car_evaluation.csv",
    header=None,
    sep=r'[,;\s]+', engine="python"
)

# Add proper column names
df.columns = ["buying","maint","doors","persons","lug_boot","safety","class"]

# Features & target
X = df.drop(columns=["class"])
y = df["class"]  # values like: unacc, acc, good, vgood

# Pipeline: one-hot encode categoricals + Random Forest
pipe = Pipeline([
    ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ("rf", RandomForestClassifier(n_estimators=200, random_state=42))
])

# Train / test
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

pipe.fit(X_tr, y_tr)
pred = pipe.predict(X_te)

print("Accuracy:", accuracy_score(y_te, pred))
print("\nReport:\n", classification_report(y_te, pred))
---------------------------------------------x------------------------------------------------------x--------------------------------------------------------------

ML6 → Tic-Tac-Toe with Reinforcement Learning



import numpy as np
import random

# ---------- a. Setting up the environment ----------
# Represent the board as a string of 9 chars: "X", "O", or " "
def empty_board():
    return [" "] * 9

def available_moves(board):
    return [i for i, v in enumerate(board) if v == " "]

def check_winner(board):
    wins = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
    for a,b,c in wins:
        if board[a] == board[b] == board[c] != " ":
            return board[a]
    return None

def game_over(board):
    return check_winner(board) or " " not in board

# ---------- b. Define the game as environment ----------
def board_to_state(board):
    return "".join(board)

# ---------- c. Q-learning model ----------
Q = {}
alpha, gamma, epsilon = 0.3, 0.9, 0.1  # learning rate, discount, exploration

def get_Q(state):
    if state not in Q:
        Q[state] = np.zeros(9)  # 9 possible moves
    return Q[state]

# ---------- d. Training ----------
for episode in range(5000):
    board = empty_board()
    player = "X"
    while not game_over(board):
        state = board_to_state(board)
        moves = available_moves(board)
        q_values = get_Q(state)
        if random.random() < epsilon:
            action = random.choice(moves)
        else:
            action = max(moves, key=lambda x: q_values[x])
        board[action] = player

        reward = 0
        if check_winner(board) == "X": reward = 1
        elif check_winner(board) == "O": reward = -1

        next_state = board_to_state(board)
        if game_over(board):
            Q[state][action] += alpha * (reward - Q[state][action])
        else:
            Q[state][action] += alpha * (reward + gamma * np.max(get_Q(next_state)) - Q[state][action])

        # switch player (opponent random)
        player = "O" if player == "X" else "X"
        if player == "O" and not game_over(board):
            move = random.choice(available_moves(board))
            board[move] = "O"

# ---------- e. Testing ----------
def play():
    board = empty_board()
    while not game_over(board):
        print("\nBoard:", np.array(board).reshape(3,3))
        moves = available_moves(board)
        state = board_to_state(board)
        action = max(moves, key=lambda x: get_Q(state)[x])
        board[action] = "X"
        if game_over(board): break
        opp_move = random.choice(available_moves(board))
        board[opp_move] = "O"
    print("\nFinal Board:", np.array(board).reshape(3,3))
    print("Winner:", check_winner(board))

play()
-------------------------------------x-----------------------------------------------------x-----------------------------------------------x-----------------------

