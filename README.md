# 🌍 Environmental Impact Estimator (ML-Powered)

An interactive **Machine Learning + Data Analytics** web application built with **Streamlit** to estimate a user’s **daily carbon footprint** based on lifestyle habits such as transportation, electricity usage, diet, and shopping behavior.

This project combines **rule-based carbon emission calculations** using emission factors with **Machine Learning predictions** trained on a synthetic lifestyle dataset to provide deeper environmental insights. Users can also explore **what-if scenarios**, view **category-wise emission breakdowns**, discover their **lifestyle segment**, and receive **personalized sustainability suggestions**.

---

## 🚀 Project Overview

The **Environmental Impact Estimator** is designed to help users understand how their daily activities contribute to carbon emissions.  
It uses:

- **Rule-based emission estimation** using real-world inspired **emission factors**
- **Machine Learning prediction** using a **Random Forest Regressor**
- **Lifestyle segmentation** using **K-Means Clustering**
- **Interactive visualizations** using **Plotly**
- **Scenario simulation** for greener lifestyle decisions

The application provides a practical demonstration of how **data science, machine learning, and sustainability analytics** can be combined into a real-world environmental intelligence tool.

---

## ✨ Key Features

### 1. Carbon Footprint Estimation
- Estimates **daily carbon emissions (kg CO₂/day)** based on:
  - Transport mode
  - Daily travel distance
  - Electricity consumption
  - Diet type
  - Meals per day
  - Shopping behavior
  - Detailed food intake (beef, chicken, vegetables)

### 2. Rule-Based Emission Calculation
- Uses an **Emission Factors Dataset** to compute emissions from different categories:
  - **Transport**
  - **Energy**
  - **Food**
  - **Shopping**

### 3. Machine Learning Prediction
- Trains a **Random Forest Regressor** on a **synthetic lifestyle dataset**
- Predicts carbon footprint based on user inputs
- Compares **ML prediction** with **rule-based estimation**

### 4. Lifestyle Segmentation
- Uses **K-Means Clustering** to classify users into:
  - **Low Impact 🌱**
  - **Moderate Impact 🌍**
  - **High Impact 🔥**

### 5. What-if Scenario Analysis
- Allows users to simulate alternative lifestyle choices such as:
  - Changing transport mode
  - Reducing travel distance
  - Lowering electricity usage
- Compares current vs scenario emissions visually

### 6. Interactive Visualizations
- **Donut chart** for category-wise carbon footprint breakdown
- **Bar chart** for current vs scenario comparison
- **Feature importance chart** for ML model insights

### 7. Personalized Sustainability Suggestions
- Provides custom recommendations based on high-emission areas:
  - Cleaner transportation options
  - Reduced electricity usage
  - Diet improvements
  - Sustainable shopping habits

---

## 🧠 Machine Learning Workflow

This project includes a complete mini-ML pipeline:

### Model Used
- **Random Forest Regressor**

### Why Random Forest?
- Handles mixed feature types well
- Works effectively with non-linear relationships
- Robust against overfitting compared to a single decision tree
- Suitable for structured tabular lifestyle data

### ML Pipeline Steps
1. Load synthetic lifestyle dataset
2. Select input features and target variable (`em_total`)
3. Apply preprocessing:
   - **One-Hot Encoding** for categorical features
   - Pass-through for numerical features
4. Train model on 80% of data
5. Evaluate on 20% holdout set
6. Predict carbon footprint for current user input
7. Display:
   - **R² Score**
   - **Mean Absolute Error (MAE)**
   - **Feature Importance**

---

## 📊 Datasets Used

### 1. `emission_factors.csv`
This dataset stores **carbon emission factors** for various activities and consumption categories.

#### Example fields:
- `Category`
- `Item`
- `Unit`
- `EmissionFactor`

#### Example categories:
- Transport
- Energy
- Food
- Shopping

---

### 2. `synthetic_lifestyles.csv`
This dataset is used to train the Machine Learning model and perform clustering.

#### Example fields:
- `transport_mode`
- `transport_km_day`
- `electricity_kwh_day`
- `diet_type`
- `meals_per_day`
- `clothes_per_month`
- `electronics_per_year`
- `beef_kg_day`
- `chicken_kg_day`
- `veggies_kg_day`
- `em_total`

---

## 🛠️ Tech Stack

### Programming Language
- **Python**

### Libraries & Frameworks
- **Streamlit** – Web application framework
- **Pandas** – Data handling and preprocessing
- **NumPy** – Numerical operations
- **Scikit-learn** – Machine Learning and clustering
- **Plotly** – Interactive data visualization

### ML Techniques Used
- **Random Forest Regression**
- **K-Means Clustering**
- **One-Hot Encoding**
- **Feature Importance Analysis**
