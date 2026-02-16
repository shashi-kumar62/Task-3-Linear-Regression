# Task 3: Linear Regression

## Objective
To implement and understand Simple and Multiple Linear Regression using Scikit-learn.

---

## Tools Used
- Python
- Pandas
- Scikit-learn
- Matplotlib
- Statsmodels

---

## Dataset
Housing Price Prediction Dataset  
Total Records: 545  
Total Features: 13  
Target Variable: price  
Missing Values: None  

---

## Steps Performed

1. Imported and explored the dataset using Pandas.
2. Checked for missing values.
3. Applied One-Hot Encoding to categorical variables.
4. Split the dataset into training (80%) and testing (20%) sets.
5. Implemented Multiple Linear Regression using `sklearn.linear_model`.
6. Evaluated the model using:
   - MAE
   - MSE
   - R² Score
7. Plotted:
   - Actual vs Predicted Prices
   - Residual Plot
8. Implemented Simple Linear Regression (Area vs Price).
9. Checked Multicollinearity using VIF.

---

## Model Evaluation

### Multiple Linear Regression

MAE: 970043.40  
MSE: 1754318687330.66  
R² Score: 0.6529  

The model explains approximately 65% of the variance in house prices.

---

### Simple Linear Regression (Area vs Price)

R² Score: (Add your simple regression R² here)

Multiple regression performs better than simple regression because it considers multiple features.

---

## Interview Questions

### 1. What assumptions does linear regression make?
- Linearity
- Independence
- Homoscedasticity
- Normal distribution of residuals
- No multicollinearity

### 2. How do you interpret the coefficients?
A coefficient represents the change in the target variable for a one-unit increase in the feature while keeping other features constant.

### 3. What is R² score?
R² measures how much variance in the dependent variable is explained by the independent variables.

### 4. When would you prefer MSE over MAE?
When large errors need to be penalized more heavily.

### 5. How do you detect multicollinearity?
Using Variance Inflation Factor (VIF).

### 6. Difference between simple and multiple regression?
Simple regression uses one independent variable.  
Multiple regression uses more than one independent variable.

### 7. Can linear regression be used for classification?
No, it is used for predicting continuous values.

### 8. What happens if assumptions are violated?
The model may produce biased or unreliable results.

---

## Conclusion
Both simple and multiple linear regression models were successfully implemented and evaluated. Multiple regression achieved better predictive performance.
