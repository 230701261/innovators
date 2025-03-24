from flask import Flask, request, render_template
import pandas as pd
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load the trained ML model
try:
    with open("budget_allocation_model.pkl", "rb") as file:
        loaded_model = pickle.load(file)
except FileNotFoundError:
    print("Error: 'budget_allocation_model.pkl' not found in the current directory.")
    exit(1)

# Function to predict budget allocation percentages
def predict_budget(industry, stage, revenue, company_age, profit_margin, employee_count, growth_rate, operating_expenses, net_profit):
    # Create a DataFrame with input data
    input_data = pd.DataFrame(
        [[industry, stage, revenue, company_age, profit_margin, employee_count, growth_rate, operating_expenses, net_profit]],
        columns=["Industry", "Stage", "Revenue", "Company_Age", "Profit_Margin", "Employee_Count", "Growth_Rate", "Operating_Expenses", "Net_Profit"]
    )
    # Predict percentages using the loaded model
    predicted_percentage = loaded_model.predict(input_data)[0]
    # Define budget categories
    budget_labels = ["Marketing_Budget", "R&D_Budget", "Operations_Cost", "Salaries", "Legal_Compliance", "Miscellaneous"]
    # Create a dictionary of category: percentage
    budget_allocation = {label: round(percentage, 2) for label, percentage in zip(budget_labels, predicted_percentage)}
    return budget_allocation

# Define the main route for both GET (display form) and POST (process form)
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            # Extract form data
            industry = request.form['industry']
            stage = request.form['stage']
            revenue = float(request.form['revenue'])
            company_age = int(request.form['company_age'])
            profit_margin = float(request.form['profit_margin'])
            employee_count = int(request.form['employee_count'])
            growth_rate = float(request.form['growth_rate'])
            operating_expenses = float(request.form['operating_expenses'])
            net_profit = float(request.form['net_profit'])

            # Predict budget allocation
            budget_allocation = predict_budget(industry, stage, revenue, company_age, profit_margin, employee_count, growth_rate, operating_expenses, net_profit)
            
            # Render the template with the prediction
            return render_template('index.html', prediction=budget_allocation)
        except ValueError as e:
            # Handle invalid input (e.g., non-numeric values)
            return render_template('index.html', prediction=None, error=f"Invalid input: {str(e)}")
        except Exception as e:
            # Handle other errors
            return render_template('index.html', prediction=None, error=f"An error occurred: {str(e)}")
    
    # For GET request, just show the form
    return render_template('index.html', prediction=None, error=None)

# Optional: Run with Flask's development server (for testing only)
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)