import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved artifacts
model = joblib.load('../models/ipl_model.pkl')
scaler = joblib.load('../models/ipl_scaler.pkl')
encoders = joblib.load('../models/ipl_encoders.pkl')
features = joblib.load('../models/ipl_features.pkl')
all_teams = joblib.load('../models/ipl_teams.pkl')
all_cities = joblib.load('../models/ipl_cities.pkl')

# Active IPL teams (filter out defunct ones)
active_teams = [
    'Mumbai Indians', 'Chennai Super Kings', 'Royal Challengers Bangalore',
    'Kolkata Knight Riders', 'Sunrisers Hyderabad', 'Delhi Capitals',
    'Punjab Kings', 'Rajasthan Royals', 'Gujarat Titans', 'Lucknow Super Giants'
]
# Keep only teams that exist in our training data
teams = sorted([t for t in active_teams if t in all_teams])

# Page config
st.set_page_config(page_title='IPL Win Predictor', page_icon='🏏', layout='wide')

st.title('🏏 IPL Live Win Predictor')
st.markdown('Predict which team will win during the **second innings chase** using logistic regression trained on 15+ seasons of IPL data.')

st.divider()

# Input form
col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Batting team (chasing)', teams)
with col2:
    bowling_team = st.selectbox('Bowling team (defending)', 
                                 [t for t in teams if t != batting_team])

city = st.selectbox('Match city', sorted(all_cities))

target = st.number_input('Target score (1st innings total + 1)', 
                          min_value=1, max_value=300, value=180)

col3, col4, col5 = st.columns(3)

with col3:
    current_score = st.number_input('Current score', min_value=0, max_value=300, value=100)
with col4:
    overs_done = st.number_input('Overs completed', min_value=0.0, max_value=19.6, 
                                   value=10.0, step=0.1,
                                   help='e.g. 10.3 means 10 overs and 3 balls')
with col5:
    wickets_fallen = st.number_input('Wickets fallen', min_value=0, max_value=10, value=3)

# Predict button
if st.button('Predict win probability', type='primary', use_container_width=True):
    # Calculate features
    runs_left = target - current_score
    balls_bowled = int(overs_done) * 6 + round((overs_done - int(overs_done)) * 10)
    balls_left = 120 - balls_bowled
    wickets_left = 10 - wickets_fallen
    crr = (current_score * 6) / balls_bowled if balls_bowled > 0 else 0
    rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0
    
    # Basic validation
    if runs_left <= 0:
        st.success(f'🎉 {batting_team} has already won!')
    elif balls_left <= 0:
        st.error(f'❌ Innings over. {bowling_team} wins!')
    elif wickets_left <= 0:
        st.error(f'❌ All out! {bowling_team} wins!')
    else:
        # Encode categorical features
        bt_enc = encoders['batting_team'].transform([batting_team])[0]
        bw_enc = encoders['bowling_team'].transform([bowling_team])[0]
        city_enc = encoders['city'].transform([city])[0]
        
        # Build input row in same order as training
        input_df = pd.DataFrame([[bt_enc, bw_enc, city_enc, runs_left, balls_left, 
                                    wickets_left, target, crr, rrr]], columns=features)
        
        # Scale
        input_scaled = scaler.transform(input_df)
        
        # Predict
        proba = model.predict_proba(input_scaled)[0]
        bat_win_pct = proba[1] * 100
        bowl_win_pct = proba[0] * 100
        
        st.divider()
        st.subheader('Prediction')
        
        # Show probabilities
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric(f'{batting_team} win %', f'{bat_win_pct:.1f}%')
        with col_b:
            st.metric(f'{bowling_team} win %', f'{bowl_win_pct:.1f}%')
        
        # Progress bar
        st.progress(bat_win_pct / 100)
        
        # Match situation summary
        st.info(f'**Match situation:** {batting_team} needs **{runs_left} runs** from **{balls_left} balls** with **{wickets_left} wickets** in hand. Required run rate: **{rrr:.2f}**')

st.divider()
st.caption('Built with Logistic Regression | Trained on IPL 2008-2024 ball-by-ball data | Accuracy: 77.5%')