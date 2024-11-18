import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pm4py
import numpy as np


# Load your production data
df = pd.read_csv('Dataset/Production_Data_with_Completion_Time.csv', sep=',')
log = pm4py.format_dataframe(df)

# KDE - Knowledge-Driven Encoding with "Completion Time (minutes)" as Span
def extract_kde_features(log, window_size=3):
    kde_features = []
    
    # Define case types based on domain knowledge, e.g., by 'Activity' or 'Resource'
    log['Case Type'] = log['Activity']  # Assuming 'Activity' defines case types here
    
    for case_id, group in log.groupby('case:concept:name'):
        case_events = group.sort_values(by='time:timestamp').reset_index(drop=True)
        
        for i in range(len(case_events)):
            event_time = case_events['time:timestamp'].iloc[i]
            case_type = case_events['Case Type'].iloc[i]
            
            start_time = event_time - pd.Timedelta(minutes=window_size)
            end_time = event_time + pd.Timedelta(minutes=window_size)
            
            # Find other cases of the same type within the time window
            similar_cases = log[
                (log['time:timestamp'] >= start_time) & 
                (log['time:timestamp'] <= end_time) & 
                (log['Case Type'] == case_type) &
                (log['case:concept:name'] != case_id)
            ]
            
            # Calculate KDE features
            similar_case_count = similar_cases['case:concept:name'].nunique()
            avg_qty_completed = similar_cases['Qty Completed'].mean() if 'Qty Completed' in similar_cases else 0
            avg_qty_rejected = similar_cases['Qty Rejected'].mean() if 'Qty Rejected' in similar_cases else 0
            
            # Use "Completion Time (minutes)" directly as the duration
            avg_duration_concurrent_cases = similar_cases['Completion Time (minutes)'].mean()
            active_resources = similar_cases['Resource'].nunique() if 'Resource' in similar_cases.columns else 0
            total_qty_completed = similar_cases['Qty Completed'].sum() if 'Qty Completed' in similar_cases else 0
            
            kde_features.append({
                'Case ID': case_id,
                'Activity': case_events['Activity'].iloc[i],
                'Similar Case Count': similar_case_count,
                'Avg Qty Completed by Similar': avg_qty_completed,
                'Avg Qty Rejected by Similar': avg_qty_rejected,
                'Avg Duration of Concurrent Cases': avg_duration_concurrent_cases,
                'Active Resources Count': active_resources,
                'Total Qty Completed by Similar': total_qty_completed,
                'Completion Time (minutes)': case_events['Completion Time (minutes)'].iloc[i]
            })
    
    return pd.DataFrame(kde_features)

# DDE - Data-Driven Encoding with "Completion Time (minutes)" as Span
def extract_dde_features(log, window_size=3):
    dde_features = []
    
    for case_id, group in log.groupby('case:concept:name'):
        case_events = group.sort_values(by='time:timestamp').reset_index(drop=True)
        
        for i in range(len(case_events)):
            event_time = case_events['time:timestamp'].iloc[i]
            
            start_time = event_time - pd.Timedelta(minutes=window_size)
            end_time = event_time + pd.Timedelta(minutes=window_size)
            
            # Find cases within the time window, disregarding case types
            overlapping_cases = log[
                (log['time:timestamp'] >= start_time) & 
                (log['time:timestamp'] <= end_time) & 
                (log['case:concept:name'] != case_id)
            ]
            
            # Calculate DDE features
            concurrent_case_count = overlapping_cases['case:concept:name'].nunique()
            avg_qty_completed = overlapping_cases['Qty Completed'].mean() if 'Qty Completed' in overlapping_cases else 0
            avg_qty_rejected = overlapping_cases['Qty Rejected'].mean() if 'Qty Rejected' in overlapping_cases else 0
            
            # Use "Completion Time (minutes)" directly as the duration
            avg_duration_concurrent_cases = overlapping_cases['Completion Time (minutes)'].mean()
            active_resources = overlapping_cases['Resource'].nunique() if 'Resource' in overlapping_cases.columns else 0
            total_qty_completed = overlapping_cases['Qty Completed'].sum() if 'Qty Completed' in overlapping_cases else 0
            
            dde_features.append({
                'Case ID': case_id,
                'Activity': case_events['Activity'].iloc[i],
                'Concurrent Case Count': concurrent_case_count,
                'Avg Qty Completed by Others': avg_qty_completed,
                'Avg Qty Rejected by Others': avg_qty_rejected,
                'Avg Duration of Concurrent Cases': avg_duration_concurrent_cases,
                'Active Resources Count': active_resources,
                'Total Qty Completed by Others': total_qty_completed,
                'Completion Time (minutes)': case_events['Completion Time (minutes)'].iloc[i]
            })
    
    return pd.DataFrame(dde_features)

# Generate and save KDE features with additional inter-case features
kde_features_df = extract_kde_features(log)
kde_features_df.to_csv('Dataset/KDE_Features.csv', index=False)
print("KDE Features saved to Dataset/KDE_Features.csv")

# Generate and save DDE features with additional inter-case features
dde_features_df = extract_dde_features(log)
dde_features_df.to_csv('Dataset/DDE_Features.csv', index=False)
print("DDE Features saved to Dataset/DDE_Features.csv")

# Define function to train and evaluate a model
def train_and_evaluate_model(features_df, feature_columns, target_column='Completion Time (minutes)', model_name='Model'):
    # Split data into training and testing sets
    X = features_df[feature_columns]
    y = features_df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Calculate RMSE for evaluation
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f'Root Mean Squared Error (RMSE) for {model_name}: {rmse:.2f}')
    
    # Print predictions vs actuals
    results = pd.DataFrame({'Predicted': y_pred, 'Actual': y_test.values})
    print(f"Predictions vs Actuals for {model_name}:")
    print(results.head())

# Train and evaluate KDE model with full set of KDE features
print("Training KDE Model...")
train_and_evaluate_model(
    kde_features_df, 
    feature_columns=[
        'Similar Case Count', 'Avg Qty Completed by Similar', 'Avg Qty Rejected by Similar',
        'Avg Duration of Concurrent Cases', 'Active Resources Count', 'Total Qty Completed by Similar'
    ],
    model_name='KDE Model'
)

# Train and evaluate DDE model with full set of DDE features
print("\nTraining DDE Model...")
train_and_evaluate_model(
    dde_features_df, 
    feature_columns=[
        'Concurrent Case Count', 'Avg Qty Completed by Others', 'Avg Qty Rejected by Others',
        'Avg Duration of Concurrent Cases', 'Active Resources Count', 'Total Qty Completed by Others'
    ],
    model_name='DDE Model'
)
