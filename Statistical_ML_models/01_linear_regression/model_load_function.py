import joblib
import pandas as pd

# Load the trained model
model = joblib.load("assets/linear_regression_model.pkl")


def predict_yearly_amount():
    """
    Ask user for inputs and predict Yearly Amount Spent
    """
    print("Enter the following customer data:")

    # Ask for user input
    avg_session_length = float(input("Avg. Session Length: "))
    time_on_app = float(input("Time on App: "))
    time_on_website = float(input("Time on Website: "))
    length_of_membership = float(input("Length of Membership: "))

    # Create a DataFrame (single row)
    input_data = pd.DataFrame([{
        "Avg. Session Length": avg_session_length,
        "Time on App": time_on_app,
        "Time on Website": time_on_website,
        "Length of Membership": length_of_membership
    }])

    # Make prediction
    predicted_amount = model.predict(input_data)[0]

    print(f"\nPredicted Yearly Amount Spent: ${predicted_amount:.2f}")
    return predicted_amount


# Example usage
predict_yearly_amount()

