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
--------------------------------------------------------------x----------------------------------------------------------------x-----------------------------------------------------------x-----------------------------
