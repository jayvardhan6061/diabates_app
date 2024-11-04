import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset and train model
@st.cache(allow_output_mutation=True)
def load_data_and_train_model():
    # Load data
    data = pd.read_csv('diabetes.csv')
    
    # Split into features and target
    X = data.drop("Outcome", axis=1)
    y = data["Outcome"]
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # Model accuracy on test data
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, accuracy

# Initialize the app
st.title("Diabetes Prediction App")
st.write("This app uses a Logistic Regression model to predict the likelihood of diabetes.")

# Load model and get accuracy
model, accuracy = load_data_and_train_model()
st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

# User input form
st.header("Enter Patient Data:")
pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glucose Level", min_value=0, max_value=200, value=120)
blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=140, value=80)
skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0, max_value=900, value=80)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5, step=0.01)
age = st.number_input("Age", min_value=1, max_value=120, value=30)

# Prediction
if st.button("Predict Diabetes"):
    # Prepare input data for prediction
    input_data = pd.DataFrame({
        "Pregnancies": [pregnancies], "Glucose": [glucose], "BloodPressure": [blood_pressure],
        "SkinThickness": [skin_thickness], "Insulin": [insulin], "BMI": [bmi],
        "DiabetesPedigreeFunction": [dpf], "Age": [age]
    })

    # Get prediction
    prediction = model.predict(input_data)[0]
    result = "Diabetes" if prediction == 1 else "No Diabetes"
    
    # Display prediction result
    st.subheader("Prediction Result")
    st.write(f"The model predicts: **{result}**")

# Display model performance metrics
st.header("Model Performance")
st.write(f"Accuracy of the model: {accuracy * 100:.2f}%")
