# =============================================================================
# COMPLETE SVM Implementation with Data Preparation
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate, KFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# Load and prepare data
train_df = pd.read_csv("train.csv")
target = 'Loan Sanction Amount (USD)'

# Handle missing values
num_cols = train_df.select_dtypes(include=['float64', 'int64']).columns
for col in num_cols:
    train_df[col].fillna(train_df[col].median(), inplace=True)

cat_cols = train_df.select_dtypes(include=['object']).columns
for col in cat_cols:
    train_df[col].fillna(train_df[col].mode()[0], inplace=True)

# Drop irrelevant columns
drop_cols = ['Customer ID', 'Name', 'Property ID', 'Location', 'Property Location']
train_df.drop(columns=drop_cols, inplace=True)

# Feature engineering
train_df['Total_Income'] = train_df['Income (USD)'] + train_df['Current Loan Expenses (USD)']
train_df['Log_Loan_Amount'] = np.log1p(train_df[target])
train_df['Log_Income'] = np.log1p(train_df['Income (USD)'])
train_df['Age_Bin'] = pd.cut(train_df['Age'], bins=[18, 30, 40, 50, 60, 100], labels=False)

# Define features and target
numeric_features = ['Age', 'Income (USD)', 'Credit Score', 'Dependents',
                    'Current Loan Expenses (USD)', 'Property Price', 'Property Age', 'Total_Income']

categorical_features = ['Gender', 'Income Stability', 'Profession', 'Type of Employment',
                        'Has Active Credit Card', 'Co-Applicant', 'Property Type']

X = train_df[numeric_features + categorical_features]
y = train_df[target]

# Split data
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

# Create preprocessing transformers
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
])

# Create the preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Step 1: Create SVM pipeline with proper preprocessing
svm_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', SVR(kernel='rbf'))  # Using RBF kernel as default
])

# Step 2: Train the SVM model
print("Training SVM model...")
svm_pipeline.fit(X_train, y_train)

# Step 3: Predict & Evaluate on Validation Set
y_val_pred_svm = svm_pipeline.predict(X_val)
mae_val_svm = mean_absolute_error(y_val, y_val_pred_svm)
mse_val_svm = mean_squared_error(y_val, y_val_pred_svm)
rmse_val_svm = np.sqrt(mse_val_svm)
r2_val_svm = r2_score(y_val, y_val_pred_svm)
adj_r2_val_svm = 1 - (1 - r2_val_svm) * (len(y_val) - 1) / (len(y_val) - X_val.shape[1] - 1)

print("\n" + "="*50)
print("SVM MODEL RESULTS")
print("="*50)
print("--- Validation Metrics (SVM) ---")
print(f"MAE: {mae_val_svm:.2f}")
print(f"MSE: {mse_val_svm:.2f}")
print(f"RMSE: {rmse_val_svm:.2f}")
print(f"R2 Score: {r2_val_svm:.4f}")
print(f"Adjusted R2: {adj_r2_val_svm:.4f}")

# Step 4: Predict & Evaluate on Test Set
y_test_pred_svm = svm_pipeline.predict(X_test)
mae_test_svm = mean_absolute_error(y_test, y_test_pred_svm)
mse_test_svm = mean_squared_error(y_test, y_test_pred_svm)
rmse_test_svm = np.sqrt(mse_test_svm)
r2_test_svm = r2_score(y_test, y_test_pred_svm)
adj_r2_test_svm = 1 - (1 - r2_test_svm) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)

print("--- Test Metrics (SVM) ---")
print(f"MAE: {mae_test_svm:.2f}")
print(f"MSE: {mse_test_svm:.2f}")
print(f"RMSE: {rmse_test_svm:.2f}")
print(f"R2 Score: {r2_test_svm:.4f}")
print(f"Adjusted R2: {adj_r2_test_svm:.4f}")

# Step 5: Hyperparameter Tuning for SVM
print("\nPerforming hyperparameter tuning for SVM...")
param_grid = {
    'regressor__C': [0.1, 1, 10],
    'regressor__gamma': ['scale', 0.1, 1],
    'regressor__kernel': ['rbf']
}

svm_grid = GridSearchCV(
    svm_pipeline,
    param_grid,
    cv=3,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)

svm_grid.fit(X_train, y_train)

print(f"Best SVM parameters: {svm_grid.best_params_}")
print(f"Best SVM R2 score: {svm_grid.best_score_:.4f}")

# Use the best model from grid search
best_svm_model = svm_grid.best_estimator_

# Evaluate best model on test set
y_test_pred_best_svm = best_svm_model.predict(X_test)
mae_test_best_svm = mean_absolute_error(y_test, y_test_pred_best_svm)
mse_test_best_svm = mean_squared_error(y_test, y_test_pred_best_svm)
rmse_test_best_svm = np.sqrt(mse_test_best_svm)
r2_test_best_svm = r2_score(y_test, y_test_pred_best_svm)
adj_r2_test_best_svm = 1 - (1 - r2_test_best_svm) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)

print("--- Best SVM Test Metrics ---")
print(f"MAE: {mae_test_best_svm:.2f}")
print(f"MSE: {mse_test_best_svm:.2f}")
print(f"RMSE: {rmse_test_best_svm:.2f}")
print(f"R2 Score: {r2_test_best_svm:.4f}")
print(f"Adjusted R2: {adj_r2_test_best_svm:.4f}")

# Step 6: Visualizations for SVM
plt.figure(figsize=(12, 5))

# Actual vs Predicted (Test Set)
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_test_pred_best_svm, alpha=0.6, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Loan Amount')
plt.ylabel('Predicted Loan Amount')
plt.title('SVM: Actual vs Predicted (Test Set)')

# Residual Plot (Test Set)
residuals_test_svm = y_test - y_test_pred_best_svm
plt.subplot(1, 2, 2)
plt.scatter(y_test_pred_best_svm, residuals_test_svm, alpha=0.6, color='purple')
plt.axhline(0, linestyle='--', color='red')
plt.xlabel('Predicted Loan Amount')
plt.ylabel('Residuals')
plt.title('SVM: Residuals vs Predicted (Test Set)')

plt.tight_layout()
plt.show()

# Step 7: Cross-Validation for SVM
print("\nPerforming cross-validation for SVM...")
kf_optimized = KFold(n_splits=3, shuffle=True, random_state=42)
scoring = {
    'MAE': 'neg_mean_absolute_error',
    'MSE': 'neg_mean_squared_error',
    'R2': 'r2'
}
cv_results_svm = cross_validate(best_svm_model, X, y, cv=kf_optimized, scoring=scoring)

# Convert to positive and create result table
mae_scores_svm = -cv_results_svm['test_MAE']
mse_scores_svm = -cv_results_svm['test_MSE']
rmse_scores_svm = np.sqrt(mse_scores_svm)
r2_scores_svm = cv_results_svm['test_R2'] 

cv_table_svm = pd.DataFrame({
    'Fold': [f'Fold {i+1}' for i in range(3)],
    'MAE': mae_scores_svm,
    'MSE': mse_scores_svm,
    'RMSE': rmse_scores_svm,
    'R2 Score': r2_scores_svm
})

cv_table_svm.loc['Average'] = cv_table_svm.drop(columns='Fold').mean()
print("\n--- SVM Cross Validation Results (3-fold) ---")
print(cv_table_svm)

print("\nSVM implementation completed successfully!")