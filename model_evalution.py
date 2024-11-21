import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import xgboost as xgb
from sklearn.impute import SimpleImputer

# Load the feature files
intra_case_features_df = pd.read_csv('Dataset/Intra_Case_Features.csv')
kde_features_df = pd.read_csv('Dataset/KDE_Features.csv')
dde_features_df = pd.read_csv('Dataset/DDE_Features.csv')
hybrid_features_df = pd.read_csv('Dataset/Hybrid_Features.csv')  # Load the precomputed hybrid features

# Function to handle missing values
def preprocess_data(features_df, feature_columns, target_column):
    X = features_df[feature_columns]
    y = features_df[target_column]
    
    # Handle missing values using SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)
    y = np.array(y)  # Ensure y is a numpy array
    
    return X, y

# Function to train model and calculate RMSE
def train_and_evaluate(features_df, feature_columns, target_column='Completion Time (minutes)', model_name='Model', model_type='random_forest'):
    X, y = preprocess_data(features_df, feature_columns, target_column)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Choose model type
    if model_type == 'random_forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == 'gradient_boosting':
        model = GradientBoostingRegressor(random_state=42)
    elif model_type == 'xgboost':
        model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    else:
        raise ValueError("Invalid model type. Choose from 'random_forest', 'gradient_boosting', or 'xgboost'.")
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Calculate RMSE for evaluation
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f'Root Mean Squared Error (RMSE) for {model_name} ({model_type.replace("_", " ").title()}): {rmse:.2f}')
    
    return rmse, y_pred, y_test

# Feature sets for each model
intra_case_features = ['Last Qty Completed', 'Last Qty Rejected']
kde_features = ['Similar Case Count', 'Avg Qty Completed by Similar', 'Avg Qty Rejected by Similar']
dde_features = ['Concurrent Case Count', 'Avg Qty Completed by Others', 'Avg Qty Rejected by Others']
hybrid_feature_columns = [
    'Last Qty Completed', 'Last Qty Rejected',
    'Similar Case Count', 'Avg Qty Completed by Similar', 'Avg Qty Rejected by Similar',
    'Avg Duration of Concurrent Cases', 'Active Resources Count', 'Total Qty Completed by Similar',
    'Concurrent Case Count', 'Avg Qty Completed by Others', 'Avg Qty Rejected by Others',
    'Avg Duration of Concurrent Cases_DDE', 'Active Resources Count_DDE', 'Total Qty Completed by Others'
]

# Evaluate Models
intra_rmse, intra_y_pred, intra_y_test = train_and_evaluate(intra_case_features_df, intra_case_features, model_name='Intra-Case Model')

kde_rmse, kde_y_pred, kde_y_test = train_and_evaluate(kde_features_df, kde_features, model_name='KDE Model')

dde_rmse, dde_y_pred, dde_y_test = train_and_evaluate(dde_features_df, dde_features, model_name='DDE Model')

# Hybrid Model (Random Forest)
hybrid_rmse_rf, hybrid_y_pred_rf, hybrid_y_test_rf = train_and_evaluate(hybrid_features_df, hybrid_feature_columns, model_name='Hybrid Model', model_type='random_forest')

# Hybrid Model (Gradient Boosting)
hybrid_rmse_gb, hybrid_y_pred_gb, hybrid_y_test_gb = train_and_evaluate(hybrid_features_df, hybrid_feature_columns, model_name='Hybrid Model', model_type='gradient_boosting')

# Hybrid Model (XGBoost)
hybrid_rmse_xgb, hybrid_y_pred_xgb, hybrid_y_test_xgb = train_and_evaluate(hybrid_features_df, hybrid_feature_columns, model_name='Hybrid Model', model_type='xgboost')

# Feature Importance Analysis for Hybrid Model (Random Forest)
hybrid_model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
X_hybrid, y_hybrid = preprocess_data(hybrid_features_df, hybrid_feature_columns, 'Completion Time (minutes)')
hybrid_model_rf.fit(X_hybrid, y_hybrid)

importances = hybrid_model_rf.feature_importances_
feature_importances = pd.DataFrame({
    'Feature': hybrid_feature_columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importance (Hybrid Model - Random Forest):")
print(feature_importances)

# Save feature importances to a CSV file
feature_importances.to_csv('Dataset/Hybrid_Feature_Importances.csv', index=False)
print("Feature Importances saved to 'Dataset/Hybrid_Feature_Importances.csv'.")

# Summarize Results
results_summary = pd.DataFrame({
    'Model': [
        'Intra-Case', 'KDE', 'DDE', 
        'Hybrid (Random Forest)', 'Hybrid (Gradient Boosting)', 'Hybrid (XGBoost)'
    ],
    'RMSE': [intra_rmse, kde_rmse, dde_rmse, hybrid_rmse_rf, hybrid_rmse_gb, hybrid_rmse_xgb]
})

print("\nSummary of Model Evaluation Results:")
print(results_summary)

# Save the results summary to a CSV file
results_summary.to_csv('Dataset/Model_Evaluation_Results.csv', index=False)

# Automatically draw conclusions
best_model_idx = results_summary['RMSE'].idxmin()
best_model_name = results_summary['Model'][best_model_idx]
best_model_rmse = results_summary['RMSE'][best_model_idx]

print(f"\nThe best performing model is '{best_model_name}' with an RMSE of {best_model_rmse:.2f}.")