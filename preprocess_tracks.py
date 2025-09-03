import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the dataset
df = pd.read_csv('tracks.csv')

# Drop duplicates
df = df.drop_duplicates()

# Handle missing values (drop rows with missing target, fill others)
df = df.dropna(subset=['popularity'])
df = df.fillna(df.median(numeric_only=True))

# Encode categorical features
label_encoders = {}
for col in ['name', 'artists']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Select numerical features to scale
num_cols = [
    'danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
    'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms'
]
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# Save preprocessed data
df.to_csv('tracks_preprocessed.csv', index=False)
print("Preprocessing complete. Saved to tracks_preprocessed.csv.")
