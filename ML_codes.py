ML1 LDA and PCA



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

ML7 : Analyzing Sales Data from Multiple File Formats

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import seaborn as sns

# ----------------------------
# 1. Load Sample Dataset from Seaborn
# ----------------------------
tips = sns.load_dataset('tips')  # built-in dataset
print("Original Dataset Head:\n", tips.head())

# Simulate multiple file formats
tips.to_csv('sample_sales.csv', index=False)
tips.to_excel('sample_sales.xlsx', index=False)
tips.to_json('sample_sales.json', orient='records')

# Load data from the simulated files
csv_data = pd.read_csv('sample_sales.csv')
excel_data = pd.read_excel('sample_sales.xlsx')
json_data = pd.read_json('sample_sales.json')

# ----------------------------
# 2. Explore Data
# ----------------------------
print("CSV Head:\n", csv_data.head())
print("Excel Head:\n", excel_data.head())
print("JSON Head:\n", json_data.head())

print("Missing Values CSV:\n", csv_data.isnull().sum())

# ----------------------------
# 3. Data Cleaning
# ----------------------------
# No missing values in this dataset for demo, but example:
csv_data.fillna(0, inplace=True)
excel_data.fillna(0, inplace=True)
json_data.fillna(0, inplace=True)

# ----------------------------
# 4. Combine Data
# ----------------------------
csv_data.columns = [col.lower() for col in csv_data.columns]
excel_data.columns = [col.lower() for col in excel_data.columns]
json_data.columns = [col.lower() for col in json_data.columns]

combined_data = pd.concat([csv_data, excel_data, json_data], ignore_index=True)

# ----------------------------
# 5. Data Transformation
# ----------------------------
# Let's treat 'total_bill' as total sales and 'tip' as profit
combined_data['total_sales'] = combined_data['total_bill']
combined_data['profit'] = combined_data['tip']

# Derive new metric: average order value (here, total_bill per customer)
combined_data['avg_order_value'] = combined_data['total_sales']

# ----------------------------
# 6. Data Analysis
# ----------------------------
print("Descriptive Statistics:\n", combined_data.describe())

# Total sales by day
total_sales_day = combined_data.groupby('day')['total_sales'].sum()
print("Total Sales by Day:\n", total_sales_day)

# Average order value
print("Average Order Value:", combined_data['avg_order_value'].mean())

# ----------------------------
# 7. Visualizations (All on one page)
# ----------------------------
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 3, figsize=(20, 6))  # 1 row, 3 columns

# Bar plot: Total sales by day
total_sales_day.plot(kind='bar', ax=axes[0], title='Total Sales by Day')
axes[0].set_xlabel('Day')
axes[0].set_ylabel('Total Sales')

# Pie chart: distribution of payment by sex
combined_data['sex'].value_counts().plot(
    kind='pie', autopct='%1.1f%%', ax=axes[1], title='Customer Gender Distribution'
)
axes[1].set_ylabel('')

# Box plot: Average order value by day
sns.boxplot(x='day', y='avg_order_value', data=combined_data, ax=axes[2])
axes[2].set_title('Average Order Value by Day')

plt.tight_layout()
plt.show()




-------------------------------------x-----------------------------------------------------x-----------------------------------------------x-----------------------
#b94eb9c1c5eaf6f822822f6abde91d24 (RN API KEY)
ML8: Analyzing Weather Data from OpenWeatherMap API
# Educational example: provide how to request the API and how to analyze results.
# NOTE: This code will not perform a real API call unless you set API_KEY and have internet access.
# Fallback: if 'weather_sample.json' exists, it will be used.


# Import libraries
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# 1. Set up API
# ----------------------------
API_KEY = ''
city = 'London'
country = 'GB'

# Base URL for current weather data
url = f"http://api.openweathermap.org/data/2.5/forecast?q={city},{country}&appid={API_KEY}&units=metric"

# ----------------------------
# 2. Retrieve Data from API
# ----------------------------
response = requests.get(url)
if response.status_code == 200:
    data = response.json()
else:
    print("Failed to fetch data:", response.status_code)
    data = None

# ----------------------------
# 3. Extract Relevant Attributes
# ----------------------------
# OpenWeatherMap 5-day forecast returns data every 3 hours under 'list'
weather_list = data['list']

# Create a DataFrame
weather_data = pd.DataFrame([{
    'datetime': item['dt_txt'],
    'temperature': item['main']['temp'],
    'temp_min': item['main']['temp_min'],
    'temp_max': item['main']['temp_max'],
    'humidity': item['main']['humidity'],
    'wind_speed': item['wind']['speed'],
    'weather_main': item['weather'][0]['main'],
    'weather_description': item['weather'][0]['description'],
    'precipitation': item.get('rain', {}).get('3h', 0)  # rain in last 3h, if available
} for item in weather_list])

# ----------------------------
# 4. Data Cleaning / Preprocessing
# ----------------------------
weather_data['datetime'] = pd.to_datetime(weather_data['datetime'])
weather_data.fillna(0, inplace=True)

# ----------------------------
# 5. Data Modeling / Analysis
# ----------------------------
# Average temperature over time
avg_temp = weather_data['temperature'].mean()
max_temp = weather_data['temp_max'].max()
min_temp = weather_data['temp_min'].min()
print(f"Average Temp: {avg_temp:.2f}°C, Max Temp: {max_temp:.2f}°C, Min Temp: {min_temp:.2f}°C")

# Aggregate daily statistics
daily_weather = weather_data.resample('D', on='datetime').agg({
    'temperature': 'mean',
    'temp_min': 'min',
    'temp_max': 'max',
    'humidity': 'mean',
    'wind_speed': 'mean',
    'precipitation': 'sum'
})
print(daily_weather)

# ----------------------------
# 6. Visualizations (All in One Page)
# ----------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f'Weather Analysis for {city}', fontsize=16, fontweight='bold')

# --- (1) Line Plot: Temperature Trends ---
axes[0, 0].plot(daily_weather.index, daily_weather['temperature'], marker='o', label='Avg Temp')
axes[0, 0].plot(daily_weather.index, daily_weather['temp_max'], marker='x', label='Max Temp')
axes[0, 0].plot(daily_weather.index, daily_weather['temp_min'], marker='^', label='Min Temp')
axes[0, 0].set_title('Temperature Trends')
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Temperature (°C)')
axes[0, 0].legend()
axes[0, 0].grid(True)

# --- (2) Bar Plot: Daily Precipitation ---
axes[0, 1].bar(daily_weather.index, daily_weather['precipitation'], color='skyblue')
axes[0, 1].set_title('Daily Precipitation (mm)')
axes[0, 1].set_xlabel('Date')
axes[0, 1].set_ylabel('Precipitation (mm)')
axes[0, 1].grid(True)

# --- (3) Scatter Plot: Temperature vs Humidity ---
sns.scatterplot(x='temperature', y='humidity', data=weather_data, ax=axes[1, 0], color='tomato')
axes[1, 0].set_title('Temperature vs Humidity')
axes[1, 0].set_xlabel('Temperature (°C)')
axes[1, 0].set_ylabel('Humidity (%)')

# --- (4) Heatmap: Correlation ---
sns.heatmap(weather_data[['temperature','temp_min','temp_max','humidity','wind_speed','precipitation']].corr(),
            annot=True, cmap='coolwarm', ax=axes[1, 1])
axes[1, 1].set_title('Correlation between Weather Attributes')

plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit title
plt.show()


# ----------------------------
# 7. Optional: Geographical Visualization
# ----------------------------
# If you want to map multiple locations, you can use city coordinates
lat = data['city']['coord']['lat']
lon = data['city']['coord']['lon']
print(f"{city} Coordinates: Latitude {lat}, Longitude {lon}")


-------------------------------------x-----------------------------------------------------x-----------------------------------------------x-----------------------


ML9 → Customer churn

# Assignment ML9: Customer Churn Analysis (Telecom)
# Educational notebook. Expects './churn.csv' or will generate a small synthetic sample.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

path = "./churn.csv"
try:
    df = pd.read_csv(path)
    print("Loaded churn dataset from", path)
except Exception as e:
    print("Could not load './churn.csv' — generating synthetic sample. Error:", e)
    n = 1000
    np.random.seed(42)
    df = pd.DataFrame({
        "tenure": np.random.randint(1, 72, n),
        "MonthlyCharges": np.random.uniform(20, 120, n),
        "TotalCharges": np.random.uniform(20, 8000, n),
        "Contract": np.random.choice(["Month-to-month","One year","Two year"], n),
        "PaymentMethod": np.random.choice(["Electronic check","Mailed check","Bank transfer"], n),
        "Churn": np.random.choice(["Yes","No"], n, p=[0.2,0.8])
    })

display(df.head())

# Preprocess: simple encoding
df["Churn_bin"] = (df["Churn"]=="Yes").astype(int)
X = pd.get_dummies(df.drop(columns=["Churn","Churn_bin"]), drop_first=True)
y = df["Churn_bin"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
probs = clf.predict_proba(X_test)[:,1]

print("Classification report:\n", classification_report(y_test, pred))
print("ROC AUC:", roc_auc_score(y_test, probs))

fpr, tpr, _ = roc_curve(y_test, probs)
plt.plot(fpr, tpr)
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Curve")
plt.show()
-----------------------------------------------x-------------------------------------------------------------x--------------------------------------------------------x---------------------------------------------


ML10 → Real estate data wrangling


# Assignment ML10: Data Wrangling on Real Estate Market
# Educational notebook. Expects './real_estate.csv' or uses a synthetic example.

import pandas as pd
import numpy as np

path = "./real_estate.csv"
try:
    df = pd.read_csv(path)
    print("Loaded real estate data from", path)
except Exception as e:
    print("Could not load './real_estate.csv' — creating a synthetic dataset. Error:", e)
    n = 500
    np.random.seed(42)
    df = pd.DataFrame({
        "price": np.random.normal(300000, 50000, n).astype(int),
        "area_sqft": np.random.normal(1500, 300, n).astype(int),
        "bedrooms": np.random.choice([1,2,3,4], n, p=[0.1,0.3,0.4,0.2]),
        "bathrooms": np.random.choice([1,2,3], n, p=[0.4,0.5,0.1]),
        "year_built": np.random.choice(range(1950,2021), n),
        "city": np.random.choice(["CityA","CityB","CityC"], n)
    })

display(df.head())

# Basic wrangling steps
print("Missing values per column:\n", df.isna().sum())
# Feature engineering: price per sqft
df["price_per_sqft"] = df["price"] / df["area_sqft"]
# Binning year_built into age groups
df["age"] = 2025 - df["year_built"]
df["age_group"] = pd.cut(df["age"], bins=[-1,10,30,60,200], labels=["new","recent","mid","old"])
display(df.head())

# Aggregations
print("\nAverage price per city:\n", df.groupby("city")["price"].mean())
print("\nMedian price_per_sqft by bedrooms:\n", df.groupby("bedrooms")["price_per_sqft"].median())

----------------------------------------------------------x----------------------------------------------------------------------------x--------------------------------------------------------------x--------------------


ML11 → AQI analysis

# Assignment ML11: Analyzing AQI Trends in a City
# Educational notebook. Expects './aqi.csv' with columns ['date','AQI'] or will create sample.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = "./aqi.csv"
try:
    df = pd.read_csv(path, parse_dates=["date"])
    print("Loaded AQI data from", path)
except Exception as e:
    print("Could not load './aqi.csv' — generating sample AQI time series. Error:", e)
    dates = pd.date_range("2024-01-01", periods=365)
    np.random.seed(42)
    df = pd.DataFrame({"date": dates, "AQI": np.clip(np.random.normal(100, 30, len(dates)), 10, 300)})

df = df.sort_values("date").set_index("date")
display(df.head())

# Daily rolling average (7-day)
df["AQI_7d"] = df["AQI"].rolling(7, min_periods=1).mean()
df["AQI_30d"] = df["AQI"].rolling(30, min_periods=1).mean()

df["AQI"].plot(alpha=0.6, title="AQI Time Series")
df["AQI_7d"].plot(label="7-day MA")
df["AQI_30d"].plot(label="30-day MA")
plt.legend()
plt.show()

------------------------------------------------------------x------------------------------------------------------------------x--------------------------------------------------------------x--------------------------
ML12 → Sales performance by region
# Assignment ML12: Sales Performance by Region
# Educational notebook. Expects './sales_region.csv' with ['Region','Product','Month','Sales'] or will create a sample.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = "./sales_region.csv"
try:
    df = pd.read_csv(path)
    print("Loaded sales by region from", path)
except Exception as e:
    print("Could not load './sales_region.csv' — generating synthetic sample. Error:", e)
    rng = pd.date_range("2024-01-01", periods=12, freq="M")
    data = []
    regions = ["North","South","East","West"]
    products = ["A","B","C"]
    np.random.seed(42)
    for r in regions:
        for p in products:
            for dt in rng:
                data.append({"Region": r, "Product": p, "Month": dt, "Sales": np.random.randint(1000,5000)})
    df = pd.DataFrame(data)

display(df.head())

# Pivot table: Regions x Month total sales
df["Month"] = pd.to_datetime(df["Month"])
pivot = df.pivot_table(index="Month", columns="Region", values="Sales", aggfunc="sum")
display(pivot.head())

pivot.plot(kind="line", figsize=(10,5), title="Monthly Sales by Region")
plt.ylabel("Sales")
plt.show()

# Top regions overall
print("Total sales by region:\n", df.groupby("Region")["Sales"].sum().sort_values(ascending=False))
