import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

# Load the precomputed intra-case, KDE, and DDE feature datasets
intra_case_features_df = pd.read_csv('Dataset/Intra_Case_Features.csv')
kde_features_df = pd.read_csv('Dataset/KDE_Features.csv')
dde_features_df = pd.read_csv('Dataset/DDE_Features.csv')

# Merge intra-case, KDE, and DDE features on 'Case ID' and 'Activity'
combined_features_df = intra_case_features_df.merge(kde_features_df, on=['Case ID', 'Activity'], how='left', suffixes=('', '_KDE'))
combined_features_df = combined_features_df.merge(dde_features_df, on=['Case ID', 'Activity'], how='left', suffixes=('', '_DDE'))

# Define the complete feature set for the hybrid model
feature_columns = [
    # Intra-Case Features
    'Last Qty Completed', 'Last Qty Rejected',
    
    # KDE Features
    'Similar Case Count', 'Avg Qty Completed by Similar', 'Avg Qty Rejected by Similar',
    'Avg Duration of Concurrent Cases', 'Active Resources Count', 'Total Qty Completed by Similar',
    
    # DDE Features
    'Concurrent Case Count', 'Avg Qty Completed by Others', 'Avg Qty Rejected by Others',
    'Avg Duration of Concurrent Cases_DDE', 'Active Resources Count_DDE', 'Total Qty Completed by Others'
]

# Target column
target_column = 'Completion Time (minutes)'

# Filter for rows where all required features are available
combined_features_df = combined_features_df.dropna(subset=feature_columns + [target_column])

# Define features (X) and target (y)
X = combined_features_df[feature_columns]
y = combined_features_df[target_column]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model on the combined features
hybrid_model = RandomForestRegressor(n_estimators=100, random_state=42)
hybrid_model.fit(X_train, y_train)

# Predict on the test set
y_pred = hybrid_model.predict(X_test)

# Calculate RMSE for evaluation
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Root Mean Squared Error (RMSE) for Hybrid Model: {rmse:.2f}')

# Print predictions vs actuals for sample
results = pd.DataFrame({'Predicted': y_pred, 'Actual': y_test.values})
print("Predictions vs Actuals for Hybrid Model:")
print(results.head())

# Optional: Save the hybrid model's feature set to CSV
combined_features_df.to_csv('Dataset/Hybrid_Features.csv', index=False)
print("Combined Hybrid Features saved to Dataset/Hybrid_Features.csv")

# Plot Feature Importance
importances = hybrid_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel("Feature Importance")
plt.title("Feature Importance for Hybrid Model")
plt.gca().invert_yaxis()  # Invert y-axis for descending order
plt.savefig('Feature_Importance_Hybrid_Model.png')
plt.show()
