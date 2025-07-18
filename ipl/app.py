import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from llm_utils import get_llm_explanation

# Page settings
st.set_page_config(page_title="IPL 2025 Win Predictor", layout="centered")
st.title("🏏 IPL 2025 WIN PREDICTOR")
st.image("https://slidechef.net/wp-content/uploads/2025/03/IPL-2025-cover-final.jpg", use_container_width=True)
st.markdown("### 🤖 Enter the match scenario to predict the winning chances!")

# Load Encoders and Model
try:
    model = pickle.load(open("model.pkl", "rb"))
    encoder = pickle.load(open("team_encoder.pkl", "rb"))
    venue_encoder = pickle.load(open("venue_encoder.pkl", "rb"))
except FileNotFoundError as e:
    st.error(f"Required file not found: {e}")
    st.stop()

teams = list(encoder.classes_)
venues = list(venue_encoder.classes_)

# Inputs: Batting and Bowling Teams
col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox("🏏 Batting Team", teams)
with col2:
    bowling_team = st.selectbox("🎯 Bowling Team", [team for team in teams if team != batting_team])

venue = st.selectbox("📍 Match Venue", venues)

# Match Stats Inputs
overs = st.slider("Overs Completed", 3.0, 20.0, step=0.1)
runs = st.number_input("🏃 Current Runs", min_value=0, max_value=300, value=50)
wickets = st.slider("Wickets Lost", 0, 10, step=1)
target = st.number_input("🎯 Target Score", min_value=1, max_value=300, value=150)

# Basic Calculations
remaining_runs = target - runs
remaining_balls = int((20 - overs) * 6)
crr = runs / overs if overs > 0 else 0
rrr = remaining_runs / (remaining_balls / 6) if remaining_balls > 0 else 0

# Encoding
batting_encoded = encoder.transform([batting_team])[0]
bowling_encoded = encoder.transform([bowling_team])[0]
encoded_venue = venue_encoder.transform([venue])[0]

# Input DataFrame
input_dict = {
    "batting_team": [batting_encoded],
    "bowling_team": [bowling_encoded],
    "venue": [encoded_venue],
    "over_ball": [overs],
    "current_score": [runs],
    "wickets": [wickets],
    "runs_left": [remaining_runs],
    "balls_left": [remaining_balls],
    "crr": [crr],
    "rrr": [rrr],
}
input_df = pd.DataFrame(input_dict)

# Predict Button
if st.button("🔮 Predict Winner"):
    if overs > 20 or runs > target or wickets > 10:
        st.warning("⚠️ Please enter valid match stats.")
    else:
        with st.spinner("🧠 Crunching the numbers..."):
            try:
                proba = model.predict_proba(input_df)[0]
                batting_prob = proba[1] * 100
                bowling_prob = proba[0] * 100

                # Show probabilities
                st.metric(label=f"Win % — {batting_team}", value=f"{batting_prob:.2f}%")
                st.metric(label=f"Win % — {bowling_team}", value=f"{bowling_prob:.2f}%")

                # Pie Chart
                fig, ax = plt.subplots()
                ax.pie([batting_prob, bowling_prob],
                       labels=[batting_team, bowling_team],
                       autopct='%1.1f%%',
                       startangle=90,
                       colors=['#FFBB00', '#0057e7'],
                       explode=(0.05, 0))
                ax.axis('equal')
                st.pyplot(fig)

                # LLM Explanation (Optional)
                try:
                    explanation = get_llm_explanation(
                        input_df, np.max(proba), batting_team, bowling_team, target, venue
                    )
                    st.subheader("📘 LLM Strategic Insight")
                    st.markdown(explanation)
                except Exception as llm_error:
                    st.warning("⚠️ LLM Insight unavailable.")

            except Exception as e:
                st.error(f"Prediction failed: {e}")
