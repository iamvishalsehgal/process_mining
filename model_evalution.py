import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the feature files
intra_case_features_df = pd.read_csv('Dataset/Intra_Case_Features.csv')
kde_features_df = pd.read_csv('Dataset/KDE_Features.csv')
dde_features_df = pd.read_csv('Dataset/DDE_Features.csv')
hybrid_features_df = pd.read_csv('Dataset/Hybrid_Features.csv')  # Load the precomputed hybrid features

# Function to train model and calculate RMSE
def train_and_evaluate(features_df, feature_columns, target_column='Completion Time (minutes)', model_name='Model'):
    X = features_df[feature_columns]
    y = features_df[target_column]
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Calculate RMSE for evaluation
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f'Root Mean Squared Error (RMSE) for {model_name}: {rmse:.2f}')
    
    return rmse, y_pred, y_test

# Evaluate Intra-Case Model
intra_rmse, intra_y_pred, intra_y_test = train_and_evaluate(
    intra_case_features_df, 
    feature_columns=['Last Qty Completed', 'Last Qty Rejected'],
    model_name='Intra-Case Model'
)

# Evaluate KDE Model
kde_rmse, kde_y_pred, kde_y_test = train_and_evaluate(
    kde_features_df, 
    feature_columns=['Similar Case Count', 'Avg Qty Completed by Similar', 'Avg Qty Rejected by Similar'],
    model_name='KDE Model'
)

# Evaluate DDE Model
dde_rmse, dde_y_pred, dde_y_test = train_and_evaluate(
    dde_features_df, 
    feature_columns=['Concurrent Case Count', 'Avg Qty Completed by Others', 'Avg Qty Rejected by Others'],
    model_name='DDE Model'
)

# Define the hybrid feature columns as used in Hybrid_Features.csv
hybrid_feature_columns = [
    # Intra-Case Features
    'Last Qty Completed', 'Last Qty Rejected',
    
    # KDE Features
    'Similar Case Count', 'Avg Qty Completed by Similar', 'Avg Qty Rejected by Similar',
    'Avg Duration of Concurrent Cases', 'Active Resources Count', 'Total Qty Completed by Similar',
    
    # DDE Features
    'Concurrent Case Count', 'Avg Qty Completed by Others', 'Avg Qty Rejected by Others',
    'Avg Duration of Concurrent Cases_DDE', 'Active Resources Count_DDE', 'Total Qty Completed by Others'
]

# Evaluate Hybrid Model
hybrid_rmse, hybrid_y_pred, hybrid_y_test = train_and_evaluate(
    hybrid_features_df, 
    feature_columns=hybrid_feature_columns,
    model_name='Hybrid Model'
)

# Feature Importance Analysis for Hybrid Model
hybrid_model = RandomForestRegressor(n_estimators=100, random_state=42)
hybrid_model.fit(hybrid_features_df[hybrid_feature_columns], hybrid_features_df['Completion Time (minutes)'])

importances = hybrid_model.feature_importances_
feature_importances = pd.DataFrame({
    'Feature': hybrid_feature_columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importance (Hybrid Model):")
print(feature_importances)

# Save feature importances to a CSV file
feature_importances.to_csv('Dataset/Hybrid_Feature_Importances.csv', index=False)
print("Feature Importances saved to 'Dataset/Hybrid_Feature_Importances.csv'.")


# Summarize results
results_summary = pd.DataFrame({
    'Model': ['Intra-Case', 'KDE', 'DDE', 'Hybrid'],
    'RMSE': [intra_rmse, kde_rmse, dde_rmse, hybrid_rmse]
})

print("\nSummary of Model Evaluation Results:")
print(results_summary)

# Save the results summary to a CSV file
results_summary.to_csv('Dataset/Model_Evaluation_Results.csv', index=False)

# Save detailed predictions vs actuals for each model
intra_results = pd.DataFrame({'Predicted': intra_y_pred, 'Actual': intra_y_test.values})
kde_results = pd.DataFrame({'Predicted': kde_y_pred, 'Actual': kde_y_test.values})
dde_results = pd.DataFrame({'Predicted': dde_y_pred, 'Actual': dde_y_test.values})
hybrid_results = pd.DataFrame({'Predicted': hybrid_y_pred, 'Actual': hybrid_y_test.values})

intra_results.to_csv('Dataset/Intra_Case_Predictions.csv', index=False)
kde_results.to_csv('Dataset/KDE_Predictions.csv', index=False)
dde_results.to_csv('Dataset/DDE_Predictions.csv', index=False)
hybrid_results.to_csv('Dataset/Hybrid_Predictions.csv', index=False)

print("\nDetailed predictions saved to CSV files.")

# Automatically draw conclusions based on the RMSE values
best_model_idx = results_summary['RMSE'].idxmin()
best_model_name = results_summary['Model'][best_model_idx]
best_model_rmse = results_summary['RMSE'][best_model_idx]

conclusions = []

# Compare each model's RMSE to determine improvements
if kde_rmse < intra_rmse:
    conclusions.append(f"KDE Model improves over the Intra-Case Model by reducing RMSE by {(intra_rmse - kde_rmse):.2f}.")
else:
    conclusions.append("KDE Model does not improve over the Intra-Case Model.")

if dde_rmse < intra_rmse:
    conclusions.append(f"DDE Model improves over the Intra-Case Model by reducing RMSE by {(intra_rmse - dde_rmse):.2f}.")
else:
    conclusions.append("DDE Model does not improve over the Intra-Case Model.")

if hybrid_rmse < min(intra_rmse, kde_rmse, dde_rmse):
    conclusions.append(f"Hybrid Model improves over all individual models with an RMSE reduction of {(min(intra_rmse, kde_rmse, dde_rmse) - hybrid_rmse):.2f} compared to the best individual model.")
else:
    conclusions.append("Hybrid Model does not outperform all individual models.")

# Conclusion about the best model
conclusions.append(f"The best performing model is the {best_model_name} with an RMSE of {best_model_rmse:.2f}.")

# Print Summary and Conclusions
print("\nSummary of Model Evaluation Results:")
print(results_summary)
print("\nConclusions:")
for conclusion in conclusions:
    print("-", conclusion)
