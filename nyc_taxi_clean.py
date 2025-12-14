import pandas as pd
import os
import kagglehub
import matplotlib.pyplot as plt
import seaborn as sns

# Download dataset from KaggleHub
path = kagglehub.dataset_download("elemento/nyc-yellow-taxi-trip-data")
print("Path to dataset files:", path)

csv_file = os.path.join(path, "yellow_tripdata_2016-03.csv")

# Load dataset into a DataFrame
df = pd.read_csv(csv_file)

# Display basic info about the dataset
print(df.info())
# Display the shape of the DataFrame
print(df.shape)
# Display statistical summary for numeric columns
print(df.describe())

# Convert pickup and dropoff time columns from string to datetime format
df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])

# Split datetime into date and time
df['pickup_date'] = df['tpep_pickup_datetime'].dt.date
df['pickup_time'] = df['tpep_pickup_datetime'].dt.time
df['dropoff_date'] = df['tpep_dropoff_datetime'].dt.date
df['dropoff_time'] = df['tpep_dropoff_datetime'].dt.time


# Remove trips with zero or extremely large distances (>90 miles)
df = df[(df['trip_distance'] > 0) & (df['trip_distance'] <= 90)]
# Remove trips with zero or extremely high fares (>850 dollars)
df = df[(df['fare_amount'] > 0) & (df['fare_amount'] < 850)]



# Drop unnecessary columns
cols_to_delete = [
    'tpep_pickup_datetime', 'tpep_dropoff_datetime', 'passenger_count',
    'RatecodeID', 'store_and_fwd_flag', 'payment_type', 'extra','VendorID',
    'mta_tax', 'tip_amount', 'tolls_amount', 'improvement_surcharge'
]

df = df.drop(columns=cols_to_delete)

# Print number of missing values per column
print("Missing values:\n", df.isnull().sum()) #There are no null values in the data
# Print number of duplicated rows
print("Number of duplicated rows:", df.duplicated().sum()) #There are no duplicated rows


# Save cleaned data
output_folder = "cleaned_data"
os.makedirs(output_folder, exist_ok=True)
cleaned_file = os.path.join(output_folder, "yellow_tripdata_2016-03_cleaned.csv")
df.to_csv(cleaned_file, index=False)
print(f"Cleaned data saved to: {cleaned_file}")


#---------------plot---------------
# Save plots
plots_folder = os.path.join(output_folder, "plots")
os.makedirs(plots_folder, exist_ok=True)

# Create boxplots for numeric columns
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
for col in numeric_cols:
    plt.figure(figsize=(6,4))
    plt.boxplot(df[col].dropna())
    plt.title(f"Boxplot for {col}")
    plt.xlabel(col)
    plot_path = os.path.join(plots_folder, f"boxplot_{col}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved plot: {plot_path}")

# Create correlation heatmap
plt.figure(figsize=(12,10))

corr = df[numeric_cols].corr()

# Create heatmap with seaborn
sns.heatmap(
    corr,
    annot=True,          
    fmt=".2f",          
    cmap='coolwarm',    
    linewidths=0.5,    
    vmin=-1, vmax=1,    
    cbar_kws={"shrink": .8}  
)

plt.title("Correlation Heatmap", fontsize=16)

plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
heatmap_path = os.path.join(plots_folder, "correlation_heatmap.png")
plt.savefig(heatmap_path)
plt.close()
print(f"Saved plot: {heatmap_path}")

