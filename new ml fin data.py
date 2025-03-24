import pandas as pd
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load dataset
fin_data = pd.read_csv(r"d:\hello_world\Cleaned_Financial_Statements.csv")

# Calculate Net Profit Margin
fin_data['Net Profit Margin'] = (fin_data['Net Income'] / fin_data['Revenue']) * 100

# Encode categorical variables
categorical_columns = ['Category']
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    fin_data[col] = le.fit_transform(fin_data[col])
    label_encoders[col] = le

# Define classify_business function
def classify_business(npm):
    if npm > 10:
        return "Healthy"
    elif npm < 5:
        return "Struggling"
    else:
        return "Moderate"

fin_data['Business Status Predicted'] = fin_data['Net Profit Margin'].apply(classify_business)

# Define features and target
X = fin_data[['Category', 'Revenue', 'Gross Profit', 'Net Income']]
y = LabelEncoder().fit_transform(fin_data['Business Status Predicted'])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Save model
with open("business_status_model.pkl", "wb") as file:
    pickle.dump(model, file)

# Load model
with open("business_status_model.pkl", "rb") as file:
    loaded_model = pickle.load(file)

# Evaluate
predictions = loaded_model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, predictions))

# Prediction function (Option 1)
def predict_business_status(category, revenue, gross_profit, net_income):
    input_data = pd.DataFrame([[category, revenue, gross_profit, net_income]], 
                              columns=['Category', 'Revenue', 'Gross Profit', 'Net Income'])
    input_data['Category'] = label_encoders['Category'].transform([category])[0]
    predicted_class = loaded_model.predict(input_data)[0]
    status_labels = {0: "Healthy", 1: "Moderate", 2: "Struggling"}
    predicted_status = status_labels[predicted_class]
    print(predicted_status)
    return predicted_status

# Example usage
example_prediction = predict_business_status("IT", 394328, 170782, 99803)