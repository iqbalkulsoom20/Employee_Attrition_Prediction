import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
try:
    data = pd.read_csv("datasets/IBM_HR_Analytics.csv")
    print("Step 1: Dataset loaded successfully")
except FileNotFoundError:
    print("File not found. Please check the path.")
    exit()

# Display basic information about the dataset
print("Dataset Overview:")
print(data.info())

# Summary statistics
print("\nSummary Statistics:")
print(data.describe())

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Visualize the distribution of the target variable (Attrition)
sns.countplot(data=data, x="Attrition")
plt.title("Attrition Count")
plt.xlabel("Attrition")
plt.ylabel("Count")
plt.show()

# Check correlations between numeric features
numeric_data = data.select_dtypes(include=["float64", "int64"])
correlation_matrix = numeric_data.corr()
if not correlation_matrix.empty:
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()
else:
    print("Correlation matrix is empty. Check numeric columns.")

# Save the cleaned dataset for further steps
data.to_csv("cleaned_data.csv", index=False)
print("Cleaned dataset saved as 'cleaned_data.csv'.")
