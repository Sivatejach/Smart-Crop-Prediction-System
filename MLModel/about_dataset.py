import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv('sample1.csv') # Change path if needed

# Show basic info
print("Dataset Info:")
print(data.info())
print("\nStatistical Summary:")
print(data.describe())
print("\nMissing Values:")
print(data.isnull().sum())

# Correlation Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

# Distribution Plots
features = ['N', 'P', 'K', 'Temperature', 'Humidity', 'pH', 'Rainfall']
for col in features:
    plt.figure(figsize=(6,4))
    sns.histplot(data[col], kde=True, color='skyblue')
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()

# Crop Frequency Count
plt.figure(figsize=(14,6))
sns.countplot(y='label', data=data, order=data['Crop'].value_counts().index)
plt.title("Crop Frequency Count")
plt.xlabel("Count")
plt.ylabel("Crop")
plt.show()

# Boxplots for Outlier Detection
for col in features:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=data[col])
    plt.title(f"Boxplot of {col}")
    plt.show()

# Pairplot of Features
sns.pairplot(data[features])
plt.show()

# Average Feature Values per Crop
avg_data = data.groupby('Crop')[features].mean().reset_index()
plt.figure(figsize=(18,8))
for i, col in enumerate(features):
    plt.subplot(2, 4, i+1)
    sns.barplot(x='Crop', y=col, data=avg_data)
    plt.xticks(rotation=90)
    plt.title(f'Average {col} per Crop')
plt.tight_layout()
plt.show()
