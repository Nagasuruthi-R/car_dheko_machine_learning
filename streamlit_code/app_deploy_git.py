import streamlit as st
import pandas as pd
import pickle
import os

current_dir = os.path.dirname(__file__)

# Load the pre-trained pipeline model from pickle file
pickle_path = os.path.join(current_dir, "../pickle_file/pipeline_model.pkl")
with open(pickle_path, "rb") as file:
    model = pickle.load(file)

# Example dataframe with existing car data (replace with your actual DataFrame)
excel_path = os.path.join(current_dir, "../preprocessed_cars/new_cleaned_entire_dataset.xlsx")
df = pd.read_excel(excel_path) # Use your actual DataFrame

st.title("Car Price Prediction App")

# Filter unique values for categorical columns
unique_car_models = df['Car_Model'].str.split().str[0].unique()
unique_fuel_types = df['Fuel_Type'].unique()
unique_transmission_types = df['Transmission_Type'].unique()
number_of_seats = [4, 5, 7, 8, 9]
unique_battery_types = df['Battery_Type'].unique()
unique_cities = df['city'].unique()
unique_insurance_periods = df['Insurance_Validity_Period'].unique()
#unique_insurance_periods = df[''].unique()

# Add a dropdown in the sidebar
# select_car_model = st.sidebar.selectbox('Select Car Model Type', options=unique_car_models)
# select_fuel_type = st.sidebar.selectbox('Select Fuel Type', options=unique_fuel_types)
# select_transmission_type = st.sidebar.selectbox('Select Transmission Type', options=unique_transmission_types)
# select_number_of_seats = st.sidebar.selectbox('Select number of seats', options = number_of_seats)
# select_Insurance_Validity_Period= st.sidebar.selectbox('Select Insurance validity period', options =unique_insurance_periods)
# select_battery_types= st.sidebar.selectbox('Select battery type', options = unique_battery_types)
# select_cities= st.sidebar.selectbox('Select cities', options = unique_cities)

# # Create input fields for custom data



st.header("Random Forest Regressors Prediction")

# Select unique categorical values from the dataframe
Car_Model = st.sidebar.selectbox("Car Model", unique_car_models)
Fuel_Type = st.sidebar.selectbox("Fuel Type", unique_fuel_types)
Transmission_Type = st.sidebar.selectbox("Transmission Type", unique_transmission_types)
Battery_Type = st.sidebar.selectbox("Battery Type", unique_battery_types)
city = st.sidebar.selectbox("City", unique_cities)
Insurance_Validity_Period = st.sidebar.selectbox("Insurance Validity Period", unique_insurance_periods)

Number_of_Seats = st.sidebar.number_input("Number of Seats", min_value=2, max_value=10, value=5)
Mileage = st.sidebar.number_input("Mileage (km/l)", min_value=0.0, max_value=50.0, value=25.10)
Engine_Capacity = st.sidebar.number_input("Engine Capacity (cc)", min_value=500, max_value=5000, value=800)
Maximum_Power = st.sidebar.number_input("Maximum Power (BHP)", min_value=30.0, max_value=500.0, value=80.03)
Torque = st.sidebar.number_input("Torque (Nm)", min_value=50.0, max_value=1000.0, value=100.0)
Wheel_Size = st.sidebar.number_input("Wheel Size (inches)", min_value=12.0, max_value=25.0, value=18.00)
Kilometers_Driven = st.sidebar.number_input("Kilometers Driven", min_value=0, max_value=500000, value=50000)
Number_of_Owners = st.sidebar.number_input("Number of Owners", min_value=1, max_value=5, value=1)
Model_Year = st.sidebar.number_input("Model Year", min_value=1990, max_value=2024, value=2021)


# # Organize input data into a DataFrame
custom_df = pd.DataFrame([{
    'city': city,
    'Model_Year': Model_Year,
    'Car_Model': Car_Model,
    'Fuel_Type': Fuel_Type,
    'Transmission_Type': Transmission_Type,
    'Battery_Type': Battery_Type,
    'Number_of_Seats': Number_of_Seats,
    'Mileage_(km/l)': Mileage,
    'Engine_Capacity': Engine_Capacity,
    'Maximum_Power': Maximum_Power,
    'Torque': Torque,
    'Wheel_Size': Wheel_Size,
    'Kilometers_Driven': Kilometers_Driven,
    'Number_of_Owners': Number_of_Owners,
    'Insurance_Validity_Period': Insurance_Validity_Period
}])

st.write("Custom input data:")
st.dataframe(custom_df)

# # Make prediction using the pipeline model
if st.button("Old Car Price Prediction Value"):
    prediction = model.predict(custom_df)
    st.success(f"Predicted Value: {prediction[0]}")

