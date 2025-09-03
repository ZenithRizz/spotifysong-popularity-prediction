import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load preprocessed data
df = pd.read_csv('tracks_preprocessed.csv')

# Basic info
print("Data shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
print("\nSummary statistics:\n", df.describe())

# Popularity distribution
plt.figure(figsize=(8, 5))
sns.histplot(df['popularity'], bins=30, kde=True, color='skyblue')
plt.title('Popularity Distribution')
plt.xlabel('Popularity')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Select only numeric columns for correlation analysis
numeric_df = df.select_dtypes(include=['number'])

# Correlation with popularity
corr = numeric_df.corr()['popularity'].sort_values(ascending=False)
print("\nFeature correlations with popularity:\n", corr)

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.show()