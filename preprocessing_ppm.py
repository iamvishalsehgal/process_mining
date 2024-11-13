import pandas as pd
import pm4py
import numpy as np

# Load the dataset
data_path = 'Dataset/Production_Analysis/Production_Data.csv'
df = pd.read_csv(data_path, sep=',')

# Remove double (or multiple) spaces in column names 
df.columns = df.columns.str.replace(r'\s+', ' ', regex=True)

# Specifying the timestamp format for clarity
timestamp_format = '%Y/%m/%d %H:%M:%S.%f'  # Format of the dataset timestamps


# Additional preprocessing done after noticing erroneous data in the Span column. Therefore all columns will be checked for errors.
# Function to identify erroneous date formats
def check_dates_for_errors(column):
    try:
        df[column] = pd.to_datetime(df[column], errors='coerce') # Coerce errors to NaT 
    except Exception as e:
        print(f"There is an error in column {column}: {e}")     
    nr_errors = df[column].isna().sum()   # Count the number of NaT values
    print(f"{nr_errors} wrong dates found in '{column}' column")


# Check 'Start Timestamp' and 'Complete Timestamp' columns
check_dates_for_errors('Start Timestamp')
check_dates_for_errors('Complete Timestamp')

# Calculate Completion (minutes) Time which is the difference between Complete Timestamp and Start Timestamp
df['Completion Time (minutes)'] = (df['Complete Timestamp'] - df['Start Timestamp']).dt.total_seconds()/60

# The span column is actually the difference between the 'Complete Timestamp' and 'Start Timestamp' columns. 
# However since it contained several errors (wrong data format, but also incorrect calculations) it was removed.
df.drop(columns=['Span'], inplace=True)

# Function to check for non-numeric values in a column and convert them to NaN
def check_numerical(column):
    df[column] = pd.to_numeric(df[column], errors='coerce')  # Convert non-numeric entries to NaN
    non_numeric = df[column].isna().sum()
    print(f"{non_numeric} non-numeric values detected in '{column}' column")
    return df[column]

# Check numerical columns
numerical_columns = ['Work Order Qty', 'Qty Completed', 'Qty Rejected', 'Qty for MRB']
for col in numerical_columns:
    df[col] = check_numerical(col)

# Modify Rework column so that it is binary (1 for 'Y', 0 for blank)
df['Rework'] = df['Rework'].apply(lambda x: 1 if x == 'Y' else 0)
print("Rework column converted to binary. 'Y' to 1, and blank to 0")

# Checking for missing values in the entire dataframe
missing_values = df.isna().sum()
print("Amount of missing values for each column")
print(missing_values)

# Renaming columns for compatibility with pm4py. It uses specific column names for 
# case identifier, the activity name and the timestamp. (Check new dataset in the Dataset folder)
log = pm4py.format_dataframe(df, case_id='Case ID', activity_key='Activity', 
                             timestamp_key='Start Timestamp', timest_format=timestamp_format)

# Getting start and end activities to test the log
start_activities = pm4py.get_start_activities(log)
end_activities = pm4py.get_end_activities(log)

#print("Start Activities:", start_activities)
#print("End Activities:", end_activities)

# Save the updated dataset
df.to_csv('Dataset/Production_Data_with_Completion_Time.csv', index=False)
