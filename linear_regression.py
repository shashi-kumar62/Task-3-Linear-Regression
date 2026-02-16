# ==========================================================
# Task 3: Linear Regression - Housing Price Prediction
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from statsmodels.stats.outliers_influence import variance_inflation_factor


# ==========================================================
# 1. Load Dataset
# ==========================================================

df = pd.read_csv("Housing.csv")

print("===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== DATASET INFO =====")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())


# ==========================================================
# 2. Preprocessing (Encoding Categorical Columns)
# ==========================================================

df_encoded = pd.get_dummies(df, drop_first=True)

print("\n===== AFTER ENCODING =====")
print(df_encoded.head())


# ==========================================================
# 3. MULTIPLE LINEAR REGRESSION
# ==========================================================

print("\n\n================ MULTIPLE LINEAR REGRESSION ================")

X = df_encoded.drop("price", axis=1)
y = df_encoded["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("MAE:", mae)
print("MSE:", mse)
print("R2 Score:", r2)

# Coefficients
coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

print("\nModel Coefficients:")
print(coefficients)

print("\nIntercept:", model.intercept_)


# ==========================================================
# 4. Plot: Actual vs Predicted
# ==========================================================

plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices (Multiple Regression)")
plt.show()


# ==========================================================
# 5. Residual Plot
# ==========================================================

residuals = y_test - y_pred

plt.figure()
plt.scatter(y_pred, residuals)
plt.axhline(y=0)
plt.xlabel("Predicted Prices")
plt.ylabel("Residuals")
plt.title("Residual Plot (Multiple Regression)")
plt.show()


# ==========================================================
# 6. SIMPLE LINEAR REGRESSION (Area vs Price)
# ==========================================================

print("\n\n================ SIMPLE LINEAR REGRESSION ================")

X_simple = df[["area"]]
y_simple = df["price"]

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_simple, y_simple, test_size=0.2, random_state=42
)

simple_model = LinearRegression()
simple_model.fit(X_train_s, y_train_s)

y_pred_s = simple_model.predict(X_test_s)

mae_s = mean_absolute_error(y_test_s, y_pred_s)
mse_s = mean_squared_error(y_test_s, y_pred_s)
r2_s = r2_score(y_test_s, y_pred_s)

print("\nSimple Model Evaluation:")
print("MAE:", mae_s)
print("MSE:", mse_s)
print("R2 Score:", r2_s)

print("Coefficient (Area):", simple_model.coef_[0])
print("Intercept:", simple_model.intercept_)


# Plot Regression Line
plt.figure()
plt.scatter(X_test_s, y_test_s)
plt.plot(X_test_s, y_pred_s)
plt.xlabel("Area")
plt.ylabel("Price")
plt.title("Simple Linear Regression (Area vs Price)")
plt.show()


# ==========================================================
# 7. MULTICOLLINEARITY CHECK (VIF)
# ==========================================================

print("\n\n================ VIF (Multicollinearity Check) ================")

X_vif = X.copy()

vif_data = pd.DataFrame()
vif_data["Feature"] = X_vif.columns
vif_data["VIF"] = [
    variance_inflation_factor(X_vif.values, i)
    for i in range(X_vif.shape[1])
]

print(vif_data)


# ==========================================================
# END OF TASK
# ==========================================================
print("\nTask Completed Successfully 🚀")
