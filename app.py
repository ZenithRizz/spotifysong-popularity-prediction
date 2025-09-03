import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load preprocessed data and train Random Forest (or load saved model)
df = pd.read_csv('tracks_preprocessed.csv')
feature_cols = [
    'name', 'artists', 'duration_ms', 'danceability', 'energy', 'loudness',
    'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
]
X = df[feature_cols]
y = df['popularity']

# Train model (or load with joblib if already saved)
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X, y)

st.title("Spotify Song Popularity Prediction")

# User input
st.header("Enter Song Features")
input_data = {}
for col in feature_cols:
    input_data[col] = st.number_input(f"{col}", value=0.0)

# Predict button
if st.button("Predict Popularity"):
    input_df = pd.DataFrame([input_data])
    prediction = rf.predict(input_df)[0]
    st.success(f"Predicted Popularity: {prediction:.2f}")