import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# ---------------------------
# Data Preprocessing Functions
# ---------------------------

# Function to preprocess data and add intra- and inter-case features
def preprocess_data(df):
    # Convert timestamps
    df['Start Timestamp'] = pd.to_datetime(df['Start Timestamp'])
    df['Complete Timestamp'] = pd.to_datetime(df['Complete Timestamp'])
    
    # Calculate duration between start and complete timestamps in minutes
    df['Duration'] = (df['Complete Timestamp'] - df['Start Timestamp']).dt.total_seconds() / 60
    
    # Intra-case feature engineering: total duration, number of activities, and unique resources per case
    intra_case_features = df.groupby('Case ID').agg(
        total_duration=('Duration', 'sum'),
        num_activities=('Activity', 'count'),
        unique_resources=('Resource', pd.Series.nunique)
    ).reset_index()
    
    # Merge intra-case features back into the original dataframe
    df = df.merge(intra_case_features, on='Case ID', how='left')
    
    # Inter-case (DDE): Use time-based proximity to calculate inter-case features
    df['concurrent_cases'] = df.apply(lambda row: count_concurrent_cases(row['Start Timestamp'], row['Complete Timestamp'], df), axis=1)
    
    # Proximity to nearest case (DDE)
    df['time_proximity'] = df.apply(lambda row: calculate_time_proximity(row['Start Timestamp'], row['Complete Timestamp'], df), axis=1)

    return df

# Function to calculate the number of concurrent cases (already implemented)
def count_concurrent_cases(case_start, case_end, df):
    return ((df['Start Timestamp'] < case_end) & (df['Complete Timestamp'] > case_start)).sum()

# Function to calculate the time proximity to other cases (DDE)
def calculate_time_proximity(case_start, case_end, df):
    # Calculate the average time difference between the current case and other cases
    other_cases = df[(df['Start Timestamp'] != case_start) & (df['Complete Timestamp'] != case_end)]
    proximity = np.abs((other_cases['Start Timestamp'] - case_start).dt.total_seconds()).mean() / 60  # Convert to minutes
    return proximity if not pd.isna(proximity) else 0

# ------------------------
# Model Training Functions
# ------------------------

# Function to train and evaluate the Random Forest model
def train_and_evaluate_model(df, use_inter_case):
    if use_inter_case:
        # Use both intra-case and inter-case features
        X = df[['total_duration', 'num_activities', 'unique_resources', 'concurrent_cases', 'time_proximity']]
    else:
        # Use only intra-case features
        X = df[['total_duration', 'num_activities', 'unique_resources']]
    
    y = df['Duration']  # We are now predicting the duration in minutes
    
    # Split the data into train (60%) and temp (40%)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    
    # Further split temp into validation (20%) and test (20%)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Initialize and train the Random Forest model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Evaluate on the validation set (during training)
    y_val_pred = rf_model.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    
    # Evaluate on the test set (after training)
    y_test_pred = rf_model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    # Return validation and test RMSE, and feature importances
    feature_importances = pd.DataFrame(rf_model.feature_importances_, index=X.columns, columns=['importance']).sort_values('importance', ascending=False)
    
    return val_rmse, test_rmse, feature_importances

# ---------------------
# Main Execution Script
# ---------------------

if __name__ == "__main__":
    # Load the dataset
    df = pd.read_csv('Dataset/Production_Analysis/Production_Data.csv')
    
    # Preprocess the data (feature engineering)
    processed_df = preprocess_data(df)
    
    # Train and evaluate intra-case only model
    val_rmse_intra, test_rmse_intra, _ = train_and_evaluate_model(processed_df, use_inter_case=False)
    print(f"Intra-case only Validation RMSE (minutes): {val_rmse_intra}")
    print(f"Intra-case only Test RMSE (minutes): {test_rmse_intra}")
    
    # Train and evaluate intra + inter-case (DDE) model
    val_rmse_intra_inter, test_rmse_intra_inter, feature_importances = train_and_evaluate_model(processed_df, use_inter_case=True)
    print(f"Intra + Inter-case (DDE) Validation RMSE (minutes): {val_rmse_intra_inter}")
    print(f"Intra + Inter-case (DDE) Test RMSE (minutes): {test_rmse_intra_inter}")
    
    # Print feature importances for intra + inter-case model
    print("Feature Importances (Intra + Inter-case DDE):")
    print(feature_importances)
    
    # Save the processed data if needed
    processed_df.to_csv('Processed_Production_Data.csv', index=False)
