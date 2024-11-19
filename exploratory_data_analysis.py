import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('Dataset/Production_Analysis/Production_Data.csv', sep=',')

# Parse timestamps
df['Start Timestamp'] = pd.to_datetime(df['Start Timestamp'], errors='coerce')
df['Complete Timestamp'] = pd.to_datetime(df['Complete Timestamp'], errors='coerce')

# Calculate event duration in minutes
df['Event Duration (minutes)'] = (df['Complete Timestamp'] - df['Start Timestamp']).dt.total_seconds() / 60

# 1. Data Overview
print("Data Overview:")
print(df.info())

# Check for missing values
print("\nMissing Values:")
print(df.isna().sum())

# Check for duplicates
duplicates = df.duplicated().sum()
print(f"\nNumber of Duplicate Rows: {duplicates}")

# 2. Descriptive Statistics
print("\nDescriptive Statistics for Numerical Columns:")
print(df.describe())

print("\nValue Counts for Categorical Columns:")
categorical_cols = ['Activity', 'Resource', 'Worker ID', 'Report Type', 'Part Desc.']
for col in categorical_cols:
    print(f"\n{col}:")
    print(df[col].value_counts())

# 3. Time-Based Analysis
# Event duration distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['Event Duration (minutes)'], bins=20, kde=True)
plt.title("Distribution of Event Durations (minutes)")
plt.xlabel("Duration (minutes)")
plt.ylabel("Frequency")
plt.show()

# Event start times by hour
plt.figure(figsize=(10, 6))
df['Start Hour'] = df['Start Timestamp'].dt.hour
sns.countplot(x='Start Hour', data=df)
plt.title("Event Start Times by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Event Count")
plt.show()

# 4. Case-Level Analysis
# Total events per case
events_per_case = df['Case ID'].value_counts()
plt.figure(figsize=(10, 6))
sns.barplot(x=events_per_case.index, y=events_per_case.values)
plt.title("Total Events per Case")
plt.xlabel("Case ID")
plt.ylabel("Event Count")
plt.show()

# Total quantities completed, rejected, and for MRB
quantities = ['Qty Completed', 'Qty Rejected', 'Qty for MRB']
plt.figure(figsize=(10, 6))
df[quantities].sum().plot(kind='bar', color=['green', 'red', 'orange'])
plt.title("Total Quantities")
plt.ylabel("Count")
plt.xlabel("Quantity Type")
plt.show()

# 5. Activity Analysis
activity_counts = df['Activity'].value_counts()
plt.figure(figsize=(10, 6))
sns.barplot(y=activity_counts.index, x=activity_counts.values, palette='viridis')
plt.title("Activity Frequency")
plt.xlabel("Count")
plt.ylabel("Activity")
plt.show()

# 6. Resource Usage
resource_counts = df['Resource'].value_counts()
plt.figure(figsize=(10, 6))
sns.barplot(y=resource_counts.index, x=resource_counts.values, palette='coolwarm')
plt.title("Resource Usage Frequency")
plt.xlabel("Count")
plt.ylabel("Resource")
plt.show()

# 7. Correlations
correlation_matrix = df[['Event Duration (minutes)', 'Qty Completed', 'Qty Rejected', 'Qty for MRB']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()
