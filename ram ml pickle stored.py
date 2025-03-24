import pickle
import numpy as np

# Load the saved Q-table
with open("trained_q_table.pkl", "rb") as f:
    q_table = pickle.load(f)

budget_categories = ['Fixed_Expenses', 'Variable_Expenses', 'Discrepancy_Expenses']
actions = [-1, 0, 1]  # Decrease, Maintain, Increase

# Predict function
def predict_budget(Fixed_Expenses, Variable_Expenses, Discrepancy_Expenses,Revenue):
    optimised_budget = {}
    for i, category in enumerate(budget_categories):
        best_action_index = np.argmax(q_table[i])
        best_action = actions[best_action_index]
        optimised_budget[category] = "Increase" if best_action == 1 else "Decrease" if best_action == -1 else "Maintain"

    return optimised_budget

# Example Test Data (Byju's scenario)
# test_data = {
#     "Fixed_Expenses": 355200000000,
#     "Variable_Expenses": 503300000000,
#     "Discrepancy_Expenses": 200000000000,
#     "Revenue": 500000000000
# }

# Get Predictions
# result = predict_budget(**test_data)

# Print Result
# print("\nFinal Optimized Budget for Given Data:")
# for category, recommendation in result.items():
#     print(f"{category}: {recommendation}")
