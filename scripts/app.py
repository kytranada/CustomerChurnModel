import streamlit as st
import pandas as pd
import keplergl as kp
from streamlit_keplergl import keplergl_static
import joblib
import plotly.graph_objects as go
from visualizations import create_churn_analysis_charts, create_demographic_analysis, get_churn_insights

st.set_page_config(layout="wide", page_title="Telco Customer Churn Prediction", page_icon="📊")

# Title and Sidebar Setup
st.markdown("<h1 style='text-align: center; color: #ff6347;'>Telco Customer Churn Prediction App</h1>",
            unsafe_allow_html=True)
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Select Step", ['View Dataset', 'Geospatial Insights', 'New Customer Prediction'])


# Load consolidated data
@st.cache_data
def load_data():
    data = pd.read_parquet('./data/processed/merged.parquet')
    return data

# Load model and associated components
@st.cache_resource
def load_model():
    model = joblib.load('./model/churn_model.pkl')
    scaler = joblib.load('./model/scaler.pkl')
    numerical_columns, categorical_columns = joblib.load('./model/columns.pkl')
    return model, scaler, numerical_columns, categorical_columns


def create_binary_indicator(probability, threshold=0.5):
    color = "green" if probability < threshold else "red"
    fig = go.Figure(go.Indicator(
        mode="number+delta",
        value=probability * 100,
        number={'suffix': "%", 'font': {'size': 70, 'color': color}},
        delta={'reference': threshold * 100, 'position': "top", 'font': {'size': 40}},
        domain={'x': [0, 1], 'y': [0, 1]}
    ))
    fig.update_layout(height=300)
    return fig


def prepare_prediction_data(input_data, numerical_columns, categorical_columns, scaler, feature_names):
    """
    Prepare the input data for prediction by ensuring correct feature ordering and preprocessing
    """
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
    st.markdown("<h2 style='text-align: center;'>📊 Consolidated Dataset</h2>", unsafe_allow_html=True)

    data = load_data()

    # Add data statistics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Customers", len(data))
        st.metric("Average Tenure", f"{data['Tenure in Months'].mean():.1f} months")
        st.metric("Average Monthly Charge", f"${data['Monthly Charge'].mean():.2f}")


    # Add visualization tabs
    tab1, tab2, tab3 = st.tabs(["📈 Churn Analysis", "👥 Demographics", "📊 Raw Data"])

    with tab1:
        st.plotly_chart(create_churn_analysis_charts(data), use_container_width=True)

        # Add key insights
        st.subheader("Key Insights")
        insights = get_churn_insights(data)
        col1, col2 = st.columns(2)
        with col1:
            st.info(
                f"Average tenure for churned customers: {insights['avg_tenure_churned']:.1f} months\n\n"
                f"Average tenure for retained customers: {insights['avg_tenure_retained']:.1f} months"
            )
        with col2:
            st.info(
                f"Average monthly charge for churned customers: ${insights['avg_charge_churned']:.2f}\n\n"
                f"Average monthly charge for retained customers: ${insights['avg_charge_retained']:.2f}"
            )

    with tab2:
        st.plotly_chart(create_demographic_analysis(data), use_container_width=True)

        # Add demographic insights
        st.subheader("Demographic Insights")
        col1, col2 = st.columns(2)
        with col1:
            st.info(
                f"Median age of churned customers: {insights['median_age_churned']:.0f} years\n\n"
                f"Median age of retained customers: {insights['median_age_retained']:.0f} years"
            )
        with col2:
            st.info("\n".join([
                f"{gender}: {rate:.1f}% churn rate"
                for gender, rate in insights['gender_churn_rates'].items()
            ]))

    with tab3:
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
elif app_mode == 'New Customer Prediction':
    st.markdown("<h2 style='text-align: center;'>🔮 Predict New Customer Churn</h2>", unsafe_allow_html=True)

    try:
        model, scaler, numerical_columns, categorical_columns = load_model()
    except FileNotFoundError:
        st.error("Please ensure the model and associated files have been trained and saved.")
        st.stop()

    data = load_data()

    # Form for new customer data
    st.subheader("Provide New Customer Information")

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

    if st.button("Predict Churn"):
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

            threshold = 0.3

            churn_prediction = churn_prob[1] > threshold
            risk_category = "High Risk" if churn_prediction else "Low Risk"
            # Create columns for displaying results
            col1, col2 = st.columns(2)

            with col1:
                # Display gauge chart
                fig = create_binary_indicator(churn_prob[1])
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Display prediction details
                st.subheader("Prediction")
                if churn_prediction:
                    st.warning("Customer is likely to leave.")
                else:
                    st.info("Customer is likely to stay.")

                # Display risk factors based on feature importance
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

        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.error("Please check the input data format and try again.")
