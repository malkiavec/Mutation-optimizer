import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import random
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Load Lottery Data
@st.cache_data
def load_data():
    return pd.read_csv("lottery_data.csv")  # Ensure you have a dataset

df = load_data()

# Sidebar: Mutation Control
st.sidebar.header("Mutation Control")
mutation_level = st.sidebar.slider("Mutation Strength", 0.0, 1.0, 0.5)

# XGBoost Filtering Option
st.sidebar.subheader("XGBoost Filtering")
use_xgboost = st.sidebar.checkbox("Enable XGBoost Filtering", True)

# Train XGBoost Model
def train_xgboost(df):
    X = df.drop(columns=["Valid"])  # Features (remove target column)
    y = df["Valid"]  # Target: 1 (Valid), 0 (Invalid)
    model = XGBClassifier()
    model.fit(X, y)
    return model

if use_xgboost:
    xgb_model = train_xgboost(df)

# Generate Mutations
def generate_mutations(draw, mutation_level):
    mutated_draw = draw.copy()
    for _ in range(int(mutation_level * len(draw))):
        mutated_draw[random.randint(0, len(draw) - 1)] = random.randint(1, 59)
    return sorted(mutated_draw)

# LSTM Model for Sequence Learning
def build_lstm():
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(6, 1)),
        Dropout(0.2),
        Dense(6, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

lstm_model = build_lstm()

# Generate and Filter Predictions
st.subheader("Generated Predictions")
selected_draw = random.sample(range(1, 60), 6)
st.write("Base Draw:", selected_draw)

mutated_draws = [generate_mutations(selected_draw, mutation_level) for _ in range(10)]
if use_xgboost:
    filtered_draws = [draw for draw in mutated_draws if xgb_model.predict([draw])[0] == 1]
else:
    filtered_draws = mutated_draws

st.write("Filtered Predictions:", filtered_draws)

# Visualization: Number Transitions
st.subheader("Transition Graph")
G = nx.DiGraph()
for draw in filtered_draws:
    for i in range(len(draw) - 1):
        G.add_edge(draw[i], draw[i + 1])

plt.figure(figsize=(8, 6))
nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray')
st.pyplot(plt)

# Heatmap Visualization
st.subheader("Number Transition Heatmap")
transition_matrix = np.zeros((59, 59))

for draw in filtered_draws:
    for i in range(len(draw) - 1):
        transition_matrix[draw[i] - 1, draw[i + 1] - 1] += 1

sns.heatmap(transition_matrix, cmap="coolwarm", linewidths=0.5)
st.pyplot(plt)

st.write("Mutation optimization completed!")
