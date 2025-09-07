with open("MutationDetectionAgent.py", "w") as f:
    f.write("""
# Copy and paste the Streamlit code here
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import random
import requests
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# API Endpoint
API_URL = "https://data.ny.gov/resource/6nbc-h7bj.json"
CSV_FILE = "lottery_data.csv"

# Function to fetch data from API and store in CSV
@st.cache_data
def fetch_and_store_data():
    try:
        response = requests.get(API_URL)
        data = response.json()
        df = pd.DataFrame(data)

        # Extract relevant columns
        df = df[['draw_date', 'winning_numbers', 'bonus']]
        df['draw_date'] = pd.to_datetime(df['draw_date'])
        df = df.sort_values(by='draw_date', ascending=True)

        # Split winning numbers into separate columns
        df[['num1', 'num2', 'num3', 'num4', 'num5', 'num6']] = df['winning_numbers'].str.split(" ", expand=True).astype(int)
        df['bonus'] = df['bonus'].astype(int)  # Convert bonus to integer

        # Save to CSV
        df.to_csv(CSV_FILE, index=False)
        return df

    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

# Load data
df = fetch_and_store_data()

# Sidebar settings
st.sidebar.header("Mutation Adjustment")
mutation_level = st.sidebar.slider("Mutation Strength", 0.0, 1.0, 0.5)

st.sidebar.subheader("XGBoost Filtering")
use_xgboost = st.sidebar.checkbox("Enable XGBoost Filtering", True)

# Train XGBoost classifier
def train_xgboost(df):
    X = df[['num1', 'num2', 'num3', 'num4', 'num5', 'num6', 'bonus']]
    X['Valid'] = 1  # Assume all past draws are valid
    y = X.pop('Valid')

    model = XGBClassifier()
    model.fit(X, y)
    return model

if use_xgboost:
    xgb_model = train_xgboost(df)

# Generate random mutations (including the bonus in sequence)
def generate_mutations(draw, bonus, mutation_level):
    full_sequence = draw + [bonus]  # Add bonus number into the sequence
    full_sequence.sort()  # Ensure it's in the correct order

    mutated_sequence = full_sequence.copy()
    for _ in range(int(mutation_level * len(full_sequence))):
        mutated_sequence[random.randint(0, len(full_sequence) - 1)] = random.randint(1, 59)
    
    return sorted(mutated_sequence)

# LSTM model for sequence learning
def build_lstm():
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(7, 1)),  # Now training on 7 numbers (6 + bonus)
        Dense(7, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

lstm_model = build_lstm()

# Generate predictions
st.subheader("Generated Predictions")
selected_row = df.iloc[-1]  # Use last known draw as a base
selected_draw = [selected_row[f'num{i}'] for i in range(1, 7)]
selected_bonus = selected_row['bonus']

st.write("Base Draw:", selected_draw, "Bonus:", selected_bonus)

mutated_sequences = [generate_mutations(selected_draw, selected_bonus, mutation_level) for _ in range(10)]

# Filter using XGBoost
if use_xgboost:
    filtered_sequences = [seq for seq in mutated_sequences if xgb_model.predict([seq])[0] == 1]
else:
    filtered_sequences = mutated_sequences

# Display results
st.write("Filtered Predictions:")
for seq in filtered_sequences:
    main_numbers, bonus_number = seq[:-1], seq[-1]  # Extract main numbers and bonus
    st.write(f"Numbers: {main_numbers}, Bonus: {bonus_number}")

# Transition Graph
st.subheader("Transition Graph")
G = nx.DiGraph()
for seq in filtered_sequences:
    for i in range(len(seq) - 1):
        G.add_edge(seq[i], seq[i + 1])

plt.figure(figsize=(8, 6))
nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray')
st.pyplot(plt)

# Heatmap
st.subheader("Number Transition Heatmap")
transition_matrix = np.zeros((59, 59))

for seq in filtered_sequences:
    for i in range(len(seq) - 1):
        transition_matrix[seq[i] - 1, seq[i + 1] - 1] += 1

sns.heatmap(transition_matrix, cmap="coolwarm", linewidths=0.5)
st.pyplot(plt)

st.write("Mutation optimization completed!")
    """)

