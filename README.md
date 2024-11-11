# 📊Customer Churn Analysis

![Project Banner](/photo.png)

## Table of Contents

- [📈 Project Overview](#-project-overview)
- [🔍 Features](#-features)
- [🛠️ Technologies Used](#️-technologies-used)
- [📁 Project Structure](#-project-structure)
- [🚀 Installation](#-installation)
- [💡 Usage](#-usage)
- [🗃️ Data Description](#️-data-description)
- [🤖 Model Training](#-model-training)
- [🖥️ Application](#️-application)
- [📜 License](#-license)

---

## 📈 Project Overview

The **Customer Churn Analysis** project aims to predict whether a customer will leave a telecommunications company (churn) based on various features such as usage patterns, demographics, and service details. By accurately predicting churn, the company can proactively address customer concerns, improve retention strategies, and enhance overall customer satisfaction.

---

## 🔍 Features

- **View Consolidated Dataset:** Explore the entire dataset with interactive metrics and visualizations.
- **Geospatial Insights:** Visualize customer distribution and churn patterns on an interactive map using Kepler.gl.
- **New Customer Prediction:** Input new customer data to predict the likelihood of churn with visual feedback.
- **Comprehensive Data Processing:** Robust data loading, cleaning, and transformation pipeline ensuring high-quality inputs for modeling.
- **Model Training & Evaluation:** Utilize XGBoost with hyperparameter tuning to build an accurate churn prediction model.
- **Interactive Visualizations:** Leverage Plotly for dynamic charts and insights into churn factors.

---

## 🛠️ Technologies Used

- **Programming Languages:** Python
- **Web Framework:** Streamlit
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn, XGBoost
- **Visualization:** Plotly, Kepler.gl

---

## 📁 Project Structure

```
telco-churn-prediction/
├── data/
│   ├── raw/
│   │   ├── services.xlsx
│   │   ├── demographics.xlsx
│   │   ├── location.xlsx
│   │   └── status.xlsx
│   └── processed/
│       └── merged.parquet
├── model/
│   ├── churn_model.pkl
│   ├── scaler.pkl
│   └── columns.pkl
├── visualizations/
│   └── __init__.py
├── scripts/
│   ├── data_processing.py
│   └── model_training.py
├── app/
│   └── streamlit_app.py
├── README.md
├── requirements.txt
└── LICENSE
```

---

## 🚀 Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/CustomerChurnModelgit
   cd CustumerChurnModel
   ```

2. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   _If `requirements.txt` is not present, install the necessary packages manually:_

   ```bash
   pip install streamlit pandas numpy scikit-learn xgboost plotly keplergl streamlit-keplergl joblib
   ```

3. **Prepare the Data**

   Ensure that the raw data files are placed in the `./data/raw/` directory as follows:

   - `services.xlsx`
   - `demographics.xlsx`
   - `location.xlsx`
   - `status.xlsx`

   _Note: Replace the placeholder data with your actual datasets._

4. **Process the Data**

   Run the data processing script to merge, clean, and save the processed data.

   ```bash
   python scripts/data_processing.py
   ```

5. **Train the Model**

   Execute the model training script to build and save the churn prediction model.

   ```bash
   python scripts/model_training.py
   ```

6. **Run the Streamlit Application**

   Launch the web application to interact with the churn prediction system.

   ```bash
   streamlit run app/streamlit_app.py
   ```

   The app will be accessible at `http://localhost:8501`.

---

## 💡 Usage

### **1. Viewing the Dataset**

- Navigate to the **View Dataset** section.
- Explore key metrics like total customers, average tenure, and monthly charges.
- Utilize the tabs to delve into churn analysis, demographic insights, or view the raw data.

### **2. Geospatial Insights**

- Access the **Geospatial Insights** section to visualize customer locations and churn patterns on an interactive map.
- Understand regional trends and identify hotspots of customer churn.

### **3. Predicting New Customer Churn**

- Go to the **New Customer Prediction** section.
- Input relevant customer details such as tenure, monthly charges, services subscribed, and demographics.
- Click on **Predict Churn** to receive a probability score and risk assessment.
- Visual indicators and key risk factors will help interpret the prediction.

---

## 🗃️ Data Description

### **Raw Datasets**

1. **services.xlsx**

   - **Columns:** Customer ID, Tenure in Months, Phone Service, Internet Service, Streaming, Monthly Charge, Total Charges

2. **demographics.xlsx**

   - **Columns:** Customer ID, Age, Gender

3. **location.xlsx**

   - **Columns:** Customer ID, City, State, Zip Code, Latitude, Longitude

4. **status.xlsx**
   - **Columns:** Customer ID, Churn Value, Churn Category, Churn Reason

### **Processed Data**

- The raw datasets are merged on `Customer ID` to form a consolidated dataset.
- Non-essential columns are dropped, and data types are appropriately set.
- Missing values are handled, and features are scaled for modeling.
- The final processed data is saved as `merged.parquet` in the `./data/processed/` directory.

---

## 🤖 Model Training

### **Algorithm**

- **XGBoost Classifier:** Used for its performance and ability to handle complex datasets.

### **Training Pipeline**

1. **Data Loading:** Processed data is loaded from `merged.parquet`.
2. **Preprocessing:**
   - Categorical variables are encoded.
   - Numerical features are scaled using `StandardScaler`.
   - Features are split into training and testing sets with stratification to maintain class balance.
3. **Hyperparameter Tuning:**
   - Utilizes `GridSearchCV` with `StratifiedKFold` for cross-validation.
   - Parameters such as `max_depth`, `learning_rate`, `n_estimators`, and others are tuned to optimize the F1 score.
4. **Evaluation:**
   - Metrics like Accuracy, Classification Report, Confusion Matrix, and Probability Distributions are analyzed.
5. **Model Saving:**
   - The best model, scaler, and column information are serialized using `joblib` and saved in the `./model/` directory.

### **Training Script**

- Located at `scripts/model.py`
- Execute using:

  ```bash
  python scripts/model_training.py
  ```

---

## 🖥️ Application

### **Streamlit Web App**

- **File:** `app/streamlit_app.py`
- **Launch Command:**

  ```bash
  streamlit run app/streamlit_app.py
  ```

### **Features:**

1. **View Dataset:**

   - Displays key metrics and interactive visualizations.
   - Tabs for churn analysis, demographics, and raw data exploration.

2. **Geospatial Insights:**

   - Interactive map showcasing customer locations and churn density.

3. **New Customer Prediction:**
   - Input form for new customer details.
   - Predicts churn probability with visual indicators and risk factors.

### **Visualization Modules**

- Located in the `visualizations/` directory.
- Contains functions to create Plotly charts and other visual elements.

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps to contribute:

1. **Fork the Repository**

   Click the **Fork** button at the top-right corner of this page.

2. **Clone the Forked Repository**

   ```bash
   git clone https://github.com/kytranada/CustumerChurnModel.git
   cd CustumerChurnModel
   ```

3. **Create a New Branch**

   ```bash
   git checkout -b feature/YourFeatureName
   ```

4. **Make Your Changes**

   Implement your feature or bug fix.

5. **Commit Your Changes**

   ```bash
   git commit -m "Add some feature"
   ```

6. **Push to the Branch**

   ```bash
   git push origin feature/YourFeatureName
   ```

7. **Create a Pull Request**

   Navigate to the repository on GitHub and click **Compare & pull request**.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
