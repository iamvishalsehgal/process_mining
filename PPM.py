import pandas as pd
import pm4py
import numpy as np

# Load the dataset
data_path = 'Dataset/Production_Analysis/Production_Data.csv'
df = pd.read_csv(data_path, sep=',')


# Specifying the timestamp format for clarity
timestamp_format = '%Y/%m/%d %H:%M:%S.%f'  # Format of the dataset timestamps

# Ensure the timestamp columns are in the correct format
df['Start Timestamp'] = pd.to_datetime(df['Start Timestamp'], format=timestamp_format)
df['Complete Timestamp'] = pd.to_datetime(df['Complete Timestamp'], format=timestamp_format)

# Calculate Completion Time which is the difference between Complete Timestamp and Start Timestamp
df['Completion Time'] = (df['Complete Timestamp'] - df['Start Timestamp']).dt.total_seconds()

# Renaming columns for compatibility with pm4py. It uses specific column names for 
# case identifier, the activity name and the timestamp. (Check new dataset in the Dataset folder)
log = pm4py.format_dataframe(df, case_id='Case ID', activity_key='Activity', 
                             timestamp_key='Start Timestamp', timest_format=timestamp_format)

# Getting start and end activities to test the log
start_activities = pm4py.get_start_activities(log)
end_activities = pm4py.get_end_activities(log)

print("Start Activities:", start_activities)
print("End Activities:", end_activities)

# Save the updated dataset
df.to_csv('Dataset/Production_Data_with_Completion_Time.csv', index=False)

