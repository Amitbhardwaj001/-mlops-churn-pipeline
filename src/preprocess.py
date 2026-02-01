import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Load raw data
DATA_PATH = "../data/raw/churn.csv"
df = pd.read_csv(DATA_PATH)

print("Initial shape:", df.shape)

# 2. Clean column names
df.columns = df.columns.str.strip()

# 3. Handle TotalCharges issue (string → numeric)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# 4. Drop missing values
df.dropna(inplace=True)

print("Shape after cleaning:", df.shape)

# 5. Encode target variable
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# 6. One-hot encode categorical features
df = pd.get_dummies(df, drop_first=True)

# 7. Split features & target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# 8. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 9. Save processed data
X_train.to_csv("../data/processed/X_train.csv", index=False)
X_test.to_csv("../data/processed/X_test.csv", index=False)
y_train.to_csv("../data/processed/y_train.csv", index=False)
y_test.to_csv("../data/processed/y_test.csv", index=False)

print("Preprocessing complete!")
print("Processed files saved in data/processed/")
