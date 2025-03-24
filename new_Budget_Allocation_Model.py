import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv(r"C:\Users\ramal\Downloads\startup_budget_dataset (_1_).csv")

# Define features and target variables
X = df.drop(columns=["Marketing_Budget", "R&D_Budget", "Operations_Cost", "Salaries", "Legal_Compliance", "Miscellaneous"])  # Features
y = df[["Marketing_Budget", "R&D_Budget", "Operations_Cost", "Salaries", "Legal_Compliance", "Miscellaneous"]]  # Targets

# Normalize target values to percentages
y = y.div(y.sum(axis=1), axis=0) * 100

# Preprocessing
categorical_features = ["Industry", "Stage"]
numerical_features = ["Revenue", "Company_Age", "Profit_Margin", "Employee_Count", "Growth_Rate", "Operating_Expenses", "Net_Profit"]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
    ("num", StandardScaler(), numerical_features)
])

# Improved model with Gradient Boosting for better efficiency
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", MultiOutputRegressor(GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=8, min_samples_split=5, random_state=42)))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Save the trained model using pickle
with open("budget_allocation_model.pkl", "wb") as file:
    pickle.dump(model, file)

# Load the saved model
with open("budget_allocation_model.pkl", "rb") as file:
    loaded_model = pickle.load(file)

# Predict and evaluate
y_pred = loaded_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.2f}%")
print(f"Model R2 Score: {r2:.4f}")

# Function to predict budget allocation percentages using the loaded model
def predict_budget(industry, stage, revenue, company_age, profit_margin, employee_count, growth_rate, operating_expenses, net_profit):
    input_data = pd.DataFrame([[industry, stage, revenue, company_age, profit_margin, employee_count, growth_rate, operating_expenses, net_profit]],
                              columns=["Industry", "Stage", "Revenue", "Company_Age", "Profit_Margin", "Employee_Count", "Growth_Rate", "Operating_Expenses", "Net_Profit"])
    predicted_percentage = loaded_model.predict(input_data)[0]
    budget_labels = ["Marketing_Budget", "R&D_Budget", "Operations_Cost", "Salaries", "Legal_Compliance", "Miscellaneous"]
    budget_allocation = {label: round(percentage, 2) for label, percentage in zip(budget_labels, predicted_percentage)}
    return budget_allocation

# Example Prediction
example_budget = predict_budget("Tech", "Growth-stage", 1000000, 5, 20, 50, 15, 500000, 200000)
print("Predicted Budget Allocation (in %):", example_budget)
