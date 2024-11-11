import streamlit as st
import pandas as pd
import keplergl as kp
from streamlit_keplergl import keplergl_static
import joblib

st.set_page_config(layout="wide", page_title="Customer Churn Analysis", page_icon="📊")

# Title and Sidebar Setup
st.markdown("<h1 style='text-align: center; color: #ff6347;'>Customer Churn Analysis </h1>",
            unsafe_allow_html=True)
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Select Step", [ 'Customer Churn Prediction', 'Geospatial Insights', 'View Dataset'])


# Load consolidated data
@st.cache_data
def load_data() -> pd.DataFrame:
    data = pd.read_parquet('./data/processed/merged.parquet')
    return data

# Load model and associated components
@st.cache_resource
def load_model() -> tuple:
    try:
        model = joblib.load('./model/churn_model.pkl')
        scaler = joblib.load('./model/scaler.pkl')
        numerical_columns, categorical_columns = joblib.load('./model/columns.pkl')
        return model, scaler, numerical_columns, categorical_columns
    except FileNotFoundError as e:
        st.error("Model files not found. Please check the paths.")
        raise e


def prepare_prediction_data(input_data, numerical_columns, categorical_columns, scaler, feature_names):

    # Create DataFrame with the correct column order
    df = pd.DataFrame(input_data, index=[0])
    df = df[feature_names]

    # Process categorical columns
    for col in categorical_columns:
        df[col] = df[col].astype('category')
        df[col] = df[col].cat.codes

    # Scale numerical columns
    df[numerical_columns] = scaler.transform(df[numerical_columns])

    return df

# View Dataset Section
if app_mode == 'View Dataset':
    st.markdown("<h2 style='text-align: center;'>📊 Dataset</h2>", unsafe_allow_html=True)

    data = load_data()

    # Add data statistics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Customers", len(data))
        st.metric("Average Tenure", f"{data['Tenure in Months'].mean():.1f} months")
        st.metric("Average Monthly Charge", f"${data['Monthly Charge'].mean():.2f}")
        average_tenure_churned = data[data['Churn Value'] == 1]['Tenure in Months'].mean()
        st.metric("Avg Tenure of Churned Customers", f"{average_tenure_churned:.1f} months")
        
    with col2:
        st.metric("Average Age", f"{data['Age'].mean():.1f} years")
        
        gender_distribution = data['Gender'].value_counts().to_dict()
        gender_distribution_str = ', '.join([f"{key}: {value}" for key, value in gender_distribution.items()])
        st.metric("Gender Distribution", gender_distribution_str)
        
        churn_rate_by_gender = data.groupby('Gender', observed=False)['Churn Value'].mean() * 100 
        churn_rate_by_gender_str = ', '.join([f"{key}: {value:.2f}%" for key, value in churn_rate_by_gender.items()])
        st.metric("Churn Rate by Gender", churn_rate_by_gender_str)

    # Display the dataset
    st.dataframe(data, height=600)

# Geospatial Insights Section
elif app_mode == 'Geospatial Insights':
    st.markdown("<h2 style='text-align: center;'>🟠 Geospatial Customer Data Insights </h2>", unsafe_allow_html=True)

    data = load_data()
    if 'Latitude' in data.columns and 'Longitude' in data.columns:

        kepler_config = {
            "version": "v1",
            "config": {
                "visState": {
                    "filters": [],
                    "layers": [
                        {
                            "id": "churn-layer",
                            "type": "hexagon",
                            "config": {
                                "dataId": "customer_data",
                                "label": "Churn Locations",
                                "color": [255, 153, 31],
                                "highlightColor": [252, 242, 26, 255],
                                "columns": {
                                    "lat": "Latitude",
                                    "lng": "Longitude"
                                },
                                "isVisible": True,
                                "visConfig": {
                                    "radius": 10,
                                    "opacity": 0.8,
                                    "colorRange": {
                                        "name": "Custom",
                                        "type": "sequential",
                                        "category": "Custom",
                                        "colors": ["#0000ff", "#ff0000"]
                                    },
                                    "coverage": 0.8
                                },
                                "hidden": False
                            },
                        }
                    ]
                }
            }
        }
        kepler_map = kp.KeplerGl(height=800, config=kepler_config)
        kepler_map.add_data(data, name='customer_data')
        keplergl_static(kepler_map)
    else:
        st.warning("The dataset does not contain geospatial columns.")

# Customer Prediction Section
elif app_mode == 'Customer Churn Prediction':
    st.markdown("<h2 style='text-align: center;'>🔮 Predict Customer Churn</h2>", unsafe_allow_html=True)

    try:
        model, scaler, numerical_columns, categorical_columns = load_model()
    except FileNotFoundError:
        st.error("Please ensure the model and associated files have been trained and saved.")
        st.stop()

    data = load_data()

    # Input fields for customer data
    col1, col2 = st.columns(2)
    with col1:
        tenure = st.number_input("Customer Tenure (in months)", min_value=0)
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, step=0.01)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, step=0.01)
        phone_service = st.selectbox("Phone Service", sorted(data['Phone Service'].unique()))
    with col2:
        internet_service = st.selectbox("Internet Service", sorted(data['Internet Service'].unique()))
        streaming = st.selectbox("Streaming Service", sorted(data['Streaming'].unique()))
        gender = st.selectbox("Gender", sorted(data['Gender'].unique()))
        age = st.number_input("Customer Age", min_value=0)

    default_threshold = 0.5

    if st.button("Predict Churn"):
        # Validate inputs
        if monthly_charges < 0 or total_charges < 0 or age < 0:
            st.error("Please enter valid positive values for charges and age.")
            st.stop()

        # Prepare input data dictionary
        input_data = {
            # Numerical columns
            'Tenure in Months': tenure,
            'Monthly Charge': monthly_charges,
            'Total Charges': total_charges,
            'Age': age,
            # Categorical columns
            'Phone Service': phone_service,
            'Internet Service': internet_service,
            'Streaming': streaming,
            'Gender': gender,
        }

        # Prepare data for prediction
        prediction_data = prepare_prediction_data(
            input_data,
            numerical_columns,
            categorical_columns,
            scaler,
            feature_names=model.feature_names_in_
        )

        # Make prediction
        try:
            churn_prob = model.predict_proba(prediction_data)[0]
            churn_probability = churn_prob[1]
            churn_probability_percentage = round(churn_probability * 100, 2)

            # Use the default threshold for prediction
            churn_prediction = churn_prob[1] > default_threshold
            risk_category = "High Risk" if churn_prediction else "Low Risk"

            # Center the prediction results
            st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
            st.subheader("Prediction")
            st.write(f"Churn Probability: {churn_probability_percentage:.2f}%")
            if churn_prediction:
                st.warning("Customer is likely to leave")
            else:
                st.info("Customer is likely to stay")
            st.markdown("</div>", unsafe_allow_html=True)

            # Display key factors influencing the prediction
            st.subheader("Key Factors")
            factors = []
            if tenure < 12:
                factors.append("Short Tenure")
            if internet_service == 'No':
                factors.append("No Internet Service")
            if monthly_charges > data['Monthly Charge'].mean():
                factors.append("Above average monthly charges")
            if factors:
                st.warning("Risk Factors: " + ", ".join(factors))
            else:
                st.info("No significant risk factors identified")

            st.markdown("</div>", unsafe_allow_html=True)

        except ValueError as ve:
            st.error(f"Value error during prediction: {str(ve)}")
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.error("Please check the input data format and try again.")
