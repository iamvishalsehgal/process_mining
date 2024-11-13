import pandas as pd
import pm4py
import numpy as np

# Sliding Window Function for Intra-Case Features
def extract_intra_case_features(log, window_size=3):
    intra_case_features = []
    
    # Group by Case ID to iterate through each case
    for case_id, group in log.groupby('case:concept:name'):
        # Sort events within a case by timestamp
        case_events = group.sort_values(by='time:timestamp').reset_index(drop=True)
        
        # Slide over the case events with the specified window size
        for i in range(len(case_events)):
            start_idx = max(0, i - window_size + 1)
            window_events = case_events.iloc[start_idx:i+1]
            
            # Extract features based on the window
            last_qty_completed = window_events['Qty Completed'].iloc[-1] if 'Qty Completed' in window_events else 0
            last_qty_rejected = window_events['Qty Rejected'].iloc[-1] if 'Qty Rejected' in window_events else 0
            activity = window_events['Activity'].iloc[-1] if 'Activity' in window_events else 0
            #start_timestamp = window_events['time:timestamp'].iloc[0]

            # Add all features to list
            intra_case_features.append({
                'Case ID': case_id,
                'Activity': activity,
                'Last Qty Completed': last_qty_completed,
                'Last Qty Rejected': last_qty_rejected,
                'Completion Time (minutes)': case_events['Completion Time (minutes)'].iloc[i]  # target variable. Should this be included?
            })
    
    # Convert to DataFrame
    return pd.DataFrame(intra_case_features)


df = pd.read_csv('Dataset/Production_Data_with_Completion_Time.csv', sep=',')
log = pm4py.format_dataframe(df)


# Apply the feature extraction
intra_case_features_df = extract_intra_case_features(log)

# Check the output
print("Intra-Case Features with Sliding Window:")
print(intra_case_features_df.head())

# Save the intra-case features dataset if needed
intra_case_features_df.to_csv('Dataset/Intra_Case_Features.csv', index=False)


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np


# intra_case_features_df = pd.read_csv('Dataset/Intra_Case_Features.csv')

# Define the features and target variable
X = intra_case_features_df[[ 'Last Qty Completed', 'Last Qty Rejected']]
y = intra_case_features_df['Completion Time (minutes)']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate RMSE for evaluation
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Root Mean Squared Error (RMSE) on test set: {rmse:.2f}')

# Print predictions vs actuals
results = pd.DataFrame({'Predicted': y_pred, 'Actual': y_test.values})
print("Predictions vs Actuals:")
print(results.head())