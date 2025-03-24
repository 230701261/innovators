import pandas as pd
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load dataset
fin_data = pd.read_csv(r"C:\Users\ramal\Downloads\Cleaned_Financial_Statements.csv")

# Encode categorical variables
categorical_columns = ['Company', 'Category']
label_encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    fin_data[col] = le.fit_transform(fin_data[col])
    label_encoders[col] = le

# Function to classify business status
def classify_business(npm):
    if npm > 10:
        return "Healthy"
    elif npm < 5:
        return "Struggling"
    else:
        return "Moderate"

fin_data['Business Status'] = fin_data['Net Profit Margin'].apply(classify_business)

# Define features and target
X = fin_data.drop(columns=['Net Profit Margin', 'Business Status'])
y = LabelEncoder().fit_transform(fin_data['Business Status'])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Save model using pickle
with open("business_status_model.pkl", "wb") as file:
    pickle.dump(model, file)

# Load model
with open("business_status_model.pkl", "rb") as file:
    loaded_model = pickle.load(file)

# Predictions and evaluation
predictions = loaded_model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, predictions))

# Function for example prediction
def predict_business_status(company, category, *features):
    input_data = pd.DataFrame([[company, category] + list(features)], 
                              columns=X.columns)
    predicted_class = loaded_model.predict(input_data)[0]
    status_labels = {0: "Healthy", 1: "Moderate", 2: "Struggling"}  # Adjust mapping if needed
    return status_labels[predicted_class]



