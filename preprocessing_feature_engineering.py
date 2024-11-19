import pandas as pd
import numpy as np


# Utility Functions
def load_data(file_path, separator=','):
    """Loads the dataset and standardizes column names."""
    df = pd.read_csv(file_path, sep=separator)
    df.columns = df.columns.str.replace(r'\s+', ' ', regex=True)  # Clean column names
    return df


def check_dates_for_errors(df, columns):
    """Ensures specified date columns are in proper datetime format."""
    for column in columns:
        try:
            df[column] = pd.to_datetime(df[column], errors='coerce')
        except Exception as e:
            print(f"Error converting column {column} to datetime: {e}")
        errors = df[column].isna().sum()
        print(f"{errors} invalid dates found in '{column}' column.")
    return df


def check_numerical(df, columns):
    """Converts non-numeric values in specified columns to NaN."""
    for column in columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')
        errors = df[column].isna().sum()
        print(f"{errors} non-numeric values found in '{column}' column.")
    return df


def preprocess_rework_column(df, column):
    """Converts the 'Rework' column to binary."""
    df[column] = df[column].apply(lambda x: 1 if x == 'Y' else 0)
    print(f"Column '{column}' converted to binary.")
    return df


def rename_case_columns(df, case_columns):
    """Renames case-level columns with a 'case:' prefix."""
    df = df.rename(columns={col: f'case:{col}' for col in case_columns})
    return df


def remove_unnecessary_columns(df, columns_to_drop):
    """Drops specified unnecessary columns."""
    df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    print(f"Columns {columns_to_drop} dropped (if they existed).")
    return df


# Feature Engineering Functions
def add_time_features(df):
    """Adds time-based derived features."""
    df['Event Start Hour'] = df['Start Timestamp'].dt.hour
    df['Event Start Day'] = df['Start Timestamp'].dt.day
    df['Event Start Month'] = df['Start Timestamp'].dt.month
    df['Event Start Day of Week'] = df['Start Timestamp'].dt.dayofweek  # Monday=0, Sunday=6
    return df


def add_gap_features(df):
    """Adds the time gap between events for each case."""
    df['Time Gap Between Events'] = df.groupby('Case ID')['Start Timestamp'].diff().dt.total_seconds() / 60
    df['Time Gap Between Events'] = df['Time Gap Between Events'].fillna(0)  # Fill NaN for the first event
    return df


def add_cumulative_features(df):
    """Adds cumulative sums for numerical columns."""
    df['Cumulative Qty Completed'] = df.groupby('Case ID')['Qty Completed'].cumsum()
    df['Cumulative Qty Rejected'] = df.groupby('Case ID')['Qty Rejected'].cumsum()
    df['Cumulative Qty for MRB'] = df.groupby('Case ID')['Qty for MRB'].cumsum()
    return df


def add_boolean_and_aggregated_features(df):
    """Adds boolean and aggregated case-level features."""
    df['Has Rejections'] = (df['Qty Rejected'] > 0).astype(int)
    df['Total Events Per Case'] = df.groupby('Case ID')['Activity'].transform('count')
    df['Unique Activities Count'] = df.groupby('Case ID')['Activity'].transform('nunique')
    df['Unique Resources Count'] = df.groupby('Case ID')['Resource'].transform('nunique')
    return df


def add_remaining_and_duration_features(df):
    """Adds event completion time, total duration, and remaining time features."""
    df['Event Completion Time (minutes)'] = (df['Complete Timestamp'] - df['Start Timestamp']).dt.total_seconds() / 60
    df['Total Duration'] = df.groupby('Case ID')['Event Completion Time (minutes)'].transform('sum')
    df['Case Remaining Time (minutes)'] = df.groupby('Case ID').apply(
        lambda group: group['Total Duration'] - group['Event Completion Time (minutes)'].cumsum().shift(fill_value=0)
    ).reset_index(drop=True)
    return df

def rename_columns_for_pm4py(df):
    """
    Renames columns to align with PM4Py conventions.

    Args:
        df (pd.DataFrame): The input DataFrame with raw column names.

    Returns:
        pd.DataFrame: DataFrame with renamed columns.
    """
    # Define the mapping for renaming columns
    rename_mapping = {
        'Case ID': 'case:concept:name',
        'Activity': 'concept:name',
        'Start Timestamp': 'time:timestamp'
    }
    
    # Rename the columns
    df.rename(columns=rename_mapping, inplace=True)
    
    # Verify renaming
    print("Renamed columns:", [col for col in rename_mapping.values() if col in df.columns])
    
    return df


# Preprocessing Pipeline
def preprocess_data(file_path, save_path):
    """Preprocesses the raw dataset and saves the processed version."""
    # Step 1: Load Data
    df = load_data(file_path)

    # Step 2: Preprocessing
    df = check_dates_for_errors(df, ['Start Timestamp', 'Complete Timestamp'])
    df = check_numerical(df, ['Work Order Qty', 'Qty Completed', 'Qty Rejected', 'Qty for MRB'])
    df = preprocess_rework_column(df, 'Rework')
    df = remove_unnecessary_columns(df, ['Span'])

    # Step 3: Feature Engineering
    df = add_time_features(df)
    df = add_gap_features(df)
    df = add_cumulative_features(df)
    df = add_boolean_and_aggregated_features(df)
    df = add_remaining_and_duration_features(df)

    # Rename case-level columns. This is necessary for PM4Py compatibility.
    # All columns that have case-level information should be renamed.
    # E.g. these attributes have the same value for an entire case.
    df = rename_case_columns(df, ['Work Order Qty', 'Part Desc.', 'Total Duration'])

    # Rename columns for PM4Py compatibility. Case ID, Activity, and Timestamp are renamed to
    # 'case:concept:name', 'concept:name', and 'time:timestamp', respectively.
    df = rename_columns_for_pm4py(df)


    # Step 4: Check Missing Values
    missing_values = df.isna().sum()
    print("Missing values for each column:")
    print(missing_values)

    # Step 5: Save Processed Dataset
    df.to_csv(save_path, index=False)
    print(f"Preprocessing complete. Processed data saved to {save_path}.")


# Run the Pipeline
data_path = 'Dataset/Production_Analysis/Production_Data.csv'
save_path = 'Dataset/Processed_Production_Data.csv'
preprocess_data(data_path, save_path)
