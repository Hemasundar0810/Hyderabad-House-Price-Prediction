import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import joblib

# Load dataset
data = pd.read_csv("housing_data.csv")

# One-hot encode categorical columns
data = pd.get_dummies(data, drop_first=True)

# Features and target
X = data.drop("price", axis=1)
y = data["price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy metric (R2 Score)
r2 = r2_score(y_test, y_pred)
print("R2 Score:", r2)

# Save model
joblib.dump(model, "linear_regression_model.joblib")
print("Model saved successfully")