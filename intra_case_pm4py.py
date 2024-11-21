import pandas as pd
import pm4py
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import re
from sklearn.metrics import root_mean_squared_error
def load_data(file_path):
    """
    Load the dataset and date columns in proper datetime format.
    
    Parameters:
    file_path (str): Path to the CSV file containing the dataset.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame with standardized column names.
    """
    df = pd.read_csv(file_path, parse_dates=['time:timestamp', 'Complete Timestamp'])
    return df

def prepare_log(df, timestamp_format):
    """
    Prepare the log for PM4Py processing by formatting the DataFrame and converting it to an event log.
    
    Parameters:
    df (pd.DataFrame): Input dataset.
    timestamp_format (str): Format of the timestamp columns in the dataset.
    
    Returns:
    pm4py.log.EventLog: PM4Py-compatible event log.
    """


    print(df.columns)
    log = pm4py.format_dataframe(
        df, case_id='Case ID', activity_key='Activity', 
        timestamp_key='Start Timestamp', timest_format=timestamp_format
    )
    log = pm4py.convert_to_event_log(log)
    return log

# Extract Features for Cases and Events. Just for testing it. Another feature extraction function will be used.
def check_and_extract_features(df, timestamp_format):
    """
    Extract features from the event log using PM4Py's feature extraction.
    
    Parameters:
    df (pd.DataFrame): Input dataset.
    timestamp_format (str): Format of the timestamp columns in the dataset.
    
    Returns:
    pd.DataFrame: DataFrame containing extracted features for analysis.
    """
    log = prepare_log(df, timestamp_format)

    # Identify start and end activities
    start_activities = pm4py.get_start_activities(log)
    end_activities = pm4py.get_end_activities(log)

    print("Start Activities:", start_activities)
    print("End Activities:", end_activities)

    # Define features to extract
    case_level_string = ['case:Part Desc.']
    event_level_string = ['concept:name', 'Report Type', 'Resource']
    event_level_numeric = ['Qty Completed', 'Qty Rejected', 'Qty for MRB']

    # Extract features
    features_df = pm4py.ml.extract_features_dataframe(
        log=log,
        str_tr_attr=case_level_string,
        str_ev_attr=event_level_string,
        num_ev_attr=event_level_numeric,
        case_id_key='case:concept:name'
    )
    return features_df

def count_events_per_case(df):
    """
    Count the number of events per case and find cases with the maximum number of events.
    
    Parameters:
    df (pd.DataFrame): Input dataset.
    
    Returns:
    None
    """
    events_per_case = df.groupby('case:concept:name').size()  # Count events per case
    max_events = events_per_case.max()  # Maximum events in a single case
    max_event_cases = events_per_case[events_per_case == max_events].index.tolist()  # Cases with maximum events

    print(f"Maximum number of events in a case: {max_events}")
    print(f"Case(s) with the maximum number of events: {max_event_cases}")


def extract_prefix_features(df, case_id_key, target_key, numerical_vars, categorical_vars, timestamp_var, max_length):
    """
    Extract prefixes for all cases and calculate dynamic remaining time.
    
    Parameters:
    log (pm4py.log.EventLog): Event log.
    case_id_key (str): Column name for the case identifier.
    target_key (str): Column name for the target variable.
    numerical_vars (list): List of numerical columns to include.
    categorical_vars (list): List of categorical columns to include.
    timestamp_var (str): Timestamp column name.
    max_length (int): Maximum prefix length to consider.
    
    Returns:
    pd.DataFrame: DataFrame containing prefix features and remaining time.
    """
    
    prefixes_with_features = []

    for length in range(1, max_length + 1):
        # Extract prefixes of a specific length
        prefix_df = pm4py.ml.get_prefixes_from_log(df, length=length, case_id_key=case_id_key)

        # Calculate remaining time dynamically
        prefix_df['Dynamic Remaining Time'] = prefix_df.groupby(case_id_key, group_keys= False).apply(
            lambda group: group['case:Total Duration'] - group['Event Completion Time (minutes)'].cumsum().shift(fill_value=0)
        ).reset_index(level=0, drop=True)

        # prefix_df['Dynamic Remaining Time'] = prefix_df.groupby(case_id_key, group_keys=False).apply(...)

        # Select relevant columns
        selected_cols = [case_id_key, 'Dynamic Remaining Time', timestamp_var] + numerical_vars + categorical_vars
        prefixes_with_features.append(prefix_df[selected_cols])

    # Combine all prefixes and remove duplicate columns
    return pd.concat(prefixes_with_features, ignore_index=True).loc[:, ~prefixes_with_features[0].columns.duplicated()]

def split_logs(log, case_id_key):
    """
    Split event log into training, testing, and validation datasets.
    60% training, 20% testing, 20% validation.
    
    Parameters:
    log (pm4py.log.EventLog): Event log.
    case_id_key (str): Column name for the case identifier.
    
    Returns:
    tuple: Training, testing, and validation logs.
    """
    train_test_log, val_log = pm4py.ml.split_train_test(log, train_percentage=0.8, case_id_key=case_id_key)
    train_log, test_log = pm4py.ml.split_train_test(train_test_log, train_percentage=0.75, case_id_key=case_id_key)
    return train_log, test_log, val_log

def get_case_with_max_events(df):
    """
    Find the case with the maximum number of events and return its ID and the count of events.

    Parameters:
    df (pd.DataFrame): Input DataFrame containing the event log. It should have a column for case IDs.

    Returns:
    tuple: Case ID with the maximum number of events and the corresponding event count.
    """
    # Count the number of events per case
    events_per_case = df.groupby('case:concept:name').size()

    # Find the case with the maximum number of events
    max_case_id = events_per_case.idxmax()  # Case ID with the max events
    max_event_count = events_per_case.max()  # Max number of events

    return max_case_id, max_event_count

def train_model(X_train, y_train):
    """
    Train a Random Forest Regressor model.
    
    Parameters:
    X_train (pd.DataFrame): Training features.
    y_train (pd.Series): Training target values.
    
    Returns:
    RandomForestRegressor: Trained model.
    """
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, X_val, y_val):
    """
    Evaluate the trained model on test and validation datasets.
    
    Parameters:
    model (RandomForestRegressor): Trained model.
    X_test (pd.DataFrame): Test features.
    y_test (pd.Series): Test target values.
    X_val (pd.DataFrame): Validation features.
    y_val (pd.Series): Validation target values.
    
    Returns:
    None
    """
    # Predictions
    y_test_pred = model.predict(X_test)
    y_val_pred = model.predict(X_val)

    # Evaluation metrics
    test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
    val_rmse = mean_squared_error(y_val, y_val_pred, squared=False)

    test_r2 = r2_score(y_test, y_test_pred)
    val_r2 = r2_score(y_val, y_val_pred)

    # Print results
    print(f"Test RMSE: {test_rmse:.2f}")
    print(f"Validation RMSE: {val_rmse:.2f}")
    print(f"Test R²: {test_r2:.2f}")
    print(f"Validation R²: {val_r2:.2f}")

def plot_feature_importance(model, feature_names):
    """
    Plot feature importance based on the trained model.
    
    Parameters:
    model (RandomForestRegressor): Trained model.
    feature_names (list): List of feature names.
    
    Returns:
    None
    """
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': model.feature_importances_}).sort_values(by='Importance', ascending=False)
    plt.figure(figsize=(12, 6))
    plt.bar(importance_df['Feature'], importance_df['Importance'])
    plt.xticks(rotation=45, ha='right')
    plt.title('Feature Importance')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.show()

def seperate_features_from_target(df, target_col):
    """
    Prepares features (X) and target (y) from the dataset by dropping the target variable from features.
    
    Parameters:
    - df: DataFrame containing the dataset.
    - target_col: Column name for the target variable.
    
    Returns:
    - X: DataFrame of features with the target variable dropped.
    - y: Series of the target variable.
    """
    y = df[target_col]  # Extract the target variable
    X = df.drop(columns=[target_col])  # Drop the target variable from features
    return X, y


def find_columns_based_on_string(df, column_string):
    """
    Finds all columns in the DataFrame matching the specified pattern.
    
    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - column_string (str): Pattern to match column names (e.g., 'event:concept:name@').
    
    Returns:
    - set: A set of matching column names.
    """
    pattern = rf'^{re.escape(column_string)}.+'
    matching_columns = [col for col in df.columns if re.match(pattern, col)]
    return set(matching_columns)

def find_common_event_columns(train_df, val_df, test_df, column_string):
    """Finds the intersection of 'event:concept:name@...' columns across datasets."""
    train_event_columns = find_columns_based_on_string(train_df, column_string)
    val_event_columns = find_columns_based_on_string(val_df, column_string)
    test_event_columns = find_columns_based_on_string(test_df, column_string)
    
    # Find common columns
    common_event_columns = train_event_columns & val_event_columns & test_event_columns
    return common_event_columns

def filter_columns_by_common_events(df, common_event_columns):
    """Filters the DataFrame to include only the common event columns."""
    # Keep non-event columns as they might be necessary (e.g., target variable)
    non_event_columns = [col for col in df.columns if not col.startswith('event:concept:name@')]
    
    # Combine non-event columns with the common event columns
    relevant_columns = non_event_columns + list(common_event_columns)
    return df[relevant_columns]

# Validate the filtered DataFrames
def validate_filtered_columns(train_df, val_df, test_df, common_event_columns):
    """Validates that the filtered DataFrames have the same common event columns."""
    assert set(train_df.columns).intersection(common_event_columns) == common_event_columns, "Train set mismatch!"
    assert set(val_df.columns).intersection(common_event_columns) == common_event_columns, "Validation set mismatch!"
    assert set(test_df.columns).intersection(common_event_columns) == common_event_columns, "Test set mismatch!"
    print("All datasets are aligned with common event columns.")

# Main Execution
if __name__ == "__main__":
    # File paths
    data_path = 'Dataset/Processed_Production_Data.csv'
    timestamp_format = '%Y/%m/%d %H:%M:%S.%f'

    # Step 1: Load data
    df = load_data(data_path)

    # Step 2: Count events per case
    count_events_per_case(df)

    # Step 4: Define relevant columns for prefix extraction
    # These numerical and categorical vars will be used for feature extraction and machine learning
    case_id_key = 'case:concept:name'
    target_key = 'Case Remaining Time (minutes)'

    
    # Include Dynamic Remaining Time as a numerical variable, it is our correct target variable
    event_numerical_vars = ['Qty Completed', 
                      'Qty Rejected', 
                      'Qty for MRB', 
                      'Dynamic Remaining Time', 
                      #'Event Completion Time (minutes)', 
                      ]
    
    case_numerical_vars = ['case:Total Duration' 
                      ]
    numerical_vars = event_numerical_vars + case_numerical_vars
    # Note: we can also encode the categorical variables to numerical values. 
    event_categorical_vars = ['concept:name', 
                        #'Resource', 
                        'Report Type']
    case_categorical_vars = ['case:Part Desc.'                  
                             ]
    categorical_vars = event_categorical_vars + case_categorical_vars
    timestamp_var = 'time:timestamp'
    #print(df.columns)

    # Step 5: Extract prefixes and dynamic remaining time
    max_case_id, max_event_count = get_case_with_max_events(df) 
    print(f"Case ID with the maximum number of events: {max_case_id}, Event Count: {max_event_count}")
    prefixes_with_selected_features_df = extract_prefix_features(
        df, case_id_key, target_key, numerical_vars, categorical_vars, timestamp_var, max_event_count
    )
    print("Prefixes with Selected Features (first 5):")
    print(prefixes_with_selected_features_df.head())

    # Step 6: Save prefixes to a CSV, and convert to log format for splitting
    # using PM4Py.
    prefixes_with_selected_features_df.to_csv('prefixes_with_features.csv', index=False)
    log = pm4py.format_dataframe(prefixes_with_selected_features_df, case_id='case:concept:name', activity_key='concept:name', 
                             timestamp_key='time:timestamp', timest_format=timestamp_format)
    log = pm4py.convert_to_event_log(log)
    # Extract features for the entire log
    all_features_df = pm4py.extract_features_dataframe(log, str_tr_attr=case_categorical_vars, str_ev_attr=event_categorical_vars,
                                                       num_ev_attr=event_numerical_vars, case_id_key='case:concept:name')
    print("All Features Data:" , all_features_df.head())
    print(all_features_df.columns)


    # Step 7: Split logs into train, test, and validation 
    # 60% training, 20% testing, 20% validation
    train_log, test_log, val_log = split_logs(log, case_id_key)
   

    # Step 8: Prepare data for modeling.
    train_df = pm4py.extract_features_dataframe(train_log, str_tr_attr=case_categorical_vars, str_ev_attr=event_categorical_vars,
                                                num_ev_attr=event_numerical_vars, case_id_key='case:concept:name')
    test_df = pm4py.extract_features_dataframe(test_log, str_tr_attr=case_categorical_vars, str_ev_attr=event_categorical_vars,
                                               num_ev_attr=event_numerical_vars, case_id_key='case:concept:name')
    val_df = pm4py.extract_features_dataframe(val_log, str_tr_attr=case_categorical_vars, str_ev_attr=event_categorical_vars,
                                              num_ev_attr=event_numerical_vars, case_id_key='case:concept:name')
    
    # Find common event columns. This is needed to align the datasets for modeling.
    # The pm4py splitting procedure splits case based, meaning some attributes may not be present in all splits.
    common_event_columns = find_common_event_columns(train_df, val_df, test_df, column_string='event:concept:name@')

    # Filter datasets based on common columns
    train_df = filter_columns_by_common_events(train_df, common_event_columns)
    val_df = filter_columns_by_common_events(val_df, common_event_columns)
    test_df = filter_columns_by_common_events(test_df, common_event_columns)
    validate_filtered_columns(train_df, val_df, test_df, common_event_columns)
    print("Train Data:" , train_df.head()) 

    target_col = 'event:Dynamic Remaining Time'
    X_train, y_train = seperate_features_from_target(train_df, target_col)
    X_test, y_test = seperate_features_from_target(test_df, target_col)
    X_val, y_val = seperate_features_from_target(val_df, target_col)

        
    # Step 9: Train model
    model = train_model(X_train, y_train)

    # Step 10: Evaluate model
    evaluate_model(model, X_test, y_test, X_val, y_val)

    # Step 11: Plot feature importance
    feature_names = X_train.columns
    plot_feature_importance(model, feature_names)


