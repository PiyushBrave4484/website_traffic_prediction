import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV

# Load dataset

@st.cache_data
def load_data():
    data = pd.read_csv('website_traffic.csv')
    return data

data = load_data()

# Title and Description
st.title("Website Traffic Prediction App")
st.write("""
This app predicts the **Conversion Rate** using different regression models.
You can select a model, view the performance metrics, and visualize the results.
""")

# Dataset Exploration
st.write("### Dataset Exploration")
st.write(data.head())

# Encode categorical variable
encoded = pd.get_dummies(data, columns=['Traffic Source'], drop_first=True)

# Handle missing values
encoded = encoded.dropna()

# Define target and features
X = encoded.drop('Conversion Rate', axis=1)
Y = encoded['Conversion Rate']

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=44)

# Model Selection
st.write("### Model Selection")
model_choice = st.selectbox('Select Model', ['Simple Linear Regression', 'Multiple Linear Regression', 'Polynomial Regression', 'Ridge Regression'])

# Simple Linear Regression
if model_choice == 'Simple Linear Regression':
    feature = st.selectbox('Select Feature', X.columns)
    X_train = X_train[[feature]]
    X_test = X_test[[feature]]
    model = LinearRegression()

# Multiple Linear Regression
elif model_choice == 'Multiple Linear Regression':
    model = LinearRegression()

# Polynomial Regression
elif model_choice == 'Polynomial Regression':
    poly = PolynomialFeatures(degree=3)
    X_train = poly.fit_transform(X_train)
    X_test = poly.transform(X_test)
    model = LinearRegression()

# Ridge Regression
elif model_choice == 'Ridge Regression':
    poly = PolynomialFeatures(degree=3)
    X_train = poly.fit_transform(X_train)
    X_test = poly.transform(X_test)
    ridge = Ridge()
    param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}
    model = GridSearchCV(ridge, param_grid, cv=5)
    model.fit(X_train, Y_train)
    best_model = model.best_estimator_

# Fit and predict
if model_choice != 'Ridge Regression':
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
else:
    Y_pred = best_model.predict(X_test)
 
# Calculate metrics
mae = mean_absolute_error(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(Y_test, Y_pred)
adjusted_r2 = 1 - (1 - r2) * (len(Y_test) - 1) / (len(Y_test) - X_test.shape[1] - 1)

# Display metrics
st.write(f"**Mean Absolute Error (MAE):** {mae}")
st.write(f"**Mean Squared Error (MSE):** {mse}")
st.write(f"**Root Mean Squared Error (RMSE):** {rmse}")
st.write(f"**R-squared (R²):** {r2}")
st.write(f"**Adjusted R-squared (Adjusted R²):** {adjusted_r2}")

# Visualization
st.write("### Actual vs. Predicted Values")
plt.figure(figsize=(10, 6))
plt.scatter(Y_test, Y_pred, alpha=0.5)
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Conversion Rate')
plt.ylabel('Predicted Conversion Rate')
plt.title(f'Actual vs Predicted Conversion Rate ({model_choice})')
st.pyplot(plt)

# Residuals
st.write("### Residuals Distribution")
residuals = Y_test - Y_pred
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, bins=30)
plt.xlabel('Residuals')
plt.title(f'Residuals Distribution ({model_choice})')
st.pyplot(plt)

# Polynomial Regression Curve (only for Polynomial Regression and Ridge Regression)
if model_choice in ['Polynomial Regression', 'Ridge Regression']:
    st.write("### Polynomial Regression Curve")
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test[:, 1], Y_test, color='blue', label='Actual Data')
    sorted_indices = np.argsort(X_test[:, 1])
    plt.plot(X_test[sorted_indices, 1], Y_pred[sorted_indices], color='red', label='Polynomial Fit')
    plt.xlabel('Feature')
    plt.ylabel('Conversion Rate')
    plt.title('Polynomial Regression Curve')
    plt.legend()
    st.pyplot(plt)

# Conclusion
st.write("### Conclusion")
st.write("**Simple Linear Regression (SLR):** The SLR model achieved an R² of 0.0417, indicating that it explains only about 4.17% of the variance in the Conversion Rate. The low R² value suggests that a single independent variable (e.g., 'Page Views') is not sufficient to capture the complexity of the relationship with the target variable. The MAE (0.0348) and RMSE (0.0668) indicate moderate prediction errors, but the model's performance is limited due to its simplicity.")

st.write("**Multiple Linear Regression (MLR):** The MLR model performed better than SLR, with an R² of 0.1191, explaining about 11.91% of the variance in the Conversion Rate. While this is an improvement over SLR, the model still leaves a significant portion of the variance unexplained, indicating room for improvement.")

st.write("**Polynomial Regression:** The Polynomial Regression model showed the best performance among the three, with an R² of 0.2669, explaining about 26.69% of the variance in the Conversion Rate. However, the negative Adjusted R² suggests potential overfitting, meaning the model may perform well on the training data but poorly on unseen data.")

st.write("**Ridge Regression with GridSearchCV:** To address the overfitting issue in Polynomial Regression, Ridge Regression was applied with hyperparameter tuning (GridSearchCV). The Ridge model achieved an R² of 0.2118, explaining 21.18% of the variance in the Conversion Rate—a slight decrease from Polynomial Regression but with better generalization. The MAE (0.0317) and RMSE (0.0606) indicate better predictive performance compared to SLR and MLR. However, the negative Adjusted R² suggests that the model may still be overfitting, despite Ridge regression reducing complexity.")
