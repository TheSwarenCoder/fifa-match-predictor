
import streamlit as st
import pandas as pd
import numpy as np
import joblib

median_values = joblib.load("median_values.pkl")
scaler = joblib.load("scaler.pkl")

lr = joblib.load("lr_model.pkl")
xgb = joblib.load("xgb_model.pkl")
mlp = joblib.load("mlp_model.pkl")

df = pd.read_pickle("football_features.pkl")

def predict_match(home_team, away_team, tournament, neutral=False):

    home_data = (
        df[
            (df["home_team"] == home_team) |
            (df["away_team"] == home_team)
        ]
        .sort_values("date")
        .iloc[-1]
    )

    away_data = (
        df[
            (df["home_team"] == away_team) |
            (df["away_team"] == away_team)
        ]
        .sort_values("date")
        .iloc[-1]
    )

    home_elo = home_data["home_elo"] if home_data["home_team"] == home_team else home_data["away_elo"]
    away_elo = away_data["home_elo"] if away_data["home_team"] == away_team else away_data["away_elo"]

    home_form = home_data["home_form"] if home_data["home_team"] == home_team else home_data["away_form"]
    away_form = away_data["home_form"] if away_data["home_team"] == away_team else away_data["away_form"]

    home_goal_diff_form = (
        home_data["home_goal_diff_form"]
        if home_data["home_team"] == home_team
        else home_data["away_goal_diff_form"]
    )

    away_goal_diff_form = (
        away_data["home_goal_diff_form"]
        if away_data["home_team"] == away_team
        else away_data["away_goal_diff_form"]
    )

    home_avg_goals_scored = (
        home_data["home_avg_goals_scored"]
        if home_data["home_team"] == home_team
        else home_data["away_avg_goals_scored"]
    )

    away_avg_goals_scored = (
        away_data["home_avg_goals_scored"]
        if away_data["home_team"] == away_team
        else away_data["away_avg_goals_scored"]
    )

    home_avg_goals_conceded = (
        home_data["home_avg_goals_conceded"]
        if home_data["home_team"] == home_team
        else home_data["away_avg_goals_conceded"]
    )

    away_avg_goals_conceded = (
        away_data["home_avg_goals_conceded"]
        if away_data["home_team"] == away_team
        else away_data["away_avg_goals_conceded"]
    )

    home_rest_days = home_data["home_rest_days"]
    away_rest_days = away_data["home_rest_days"]

    tournament_mapping = dict(
        zip(
            df["tournament_group"],
            df["tournament_code"]
        )
    )

    tournament_code = tournament_mapping.get(
        tournament,
        0
    )

    elo_diff = home_elo - away_elo
    elo_gap_abs = abs(elo_diff)

    match_features = pd.DataFrame([{
        "home_elo": home_elo,
        "away_elo": away_elo,
        "elo_diff": elo_diff,
        "elo_gap_abs": elo_gap_abs,
        "home_form": home_form,
        "away_form": away_form,
        "home_goal_diff_form": home_goal_diff_form,
        "away_goal_diff_form": away_goal_diff_form,
        "home_avg_goals_scored": home_avg_goals_scored,
        "away_avg_goals_scored": away_avg_goals_scored,
        "home_avg_goals_conceded": home_avg_goals_conceded,
        "away_avg_goals_conceded": away_avg_goals_conceded,
        "home_rest_days": home_rest_days,
        "away_rest_days": away_rest_days,
        "tournament_code": tournament_code,
        "neutral": int(neutral)
    }])

    # Fill any missing values
    match_features = match_features.fillna(median_values)

    # Scale features
    match_features_scaled = scaler.transform(match_features)

    lr_probs = lr.predict_proba(match_features)
    xgb_probs = xgb.predict_proba(match_features)
    mlp_probs = mlp.predict_proba(match_features)

    ensemble_probs = (
        0.4 * lr_probs +
        0.3 * xgb_probs +
        0.3 * mlp_probs
    )

    prediction = np.argmax(ensemble_probs, axis=1)[0]

    return {
        "home_win": ensemble_probs[0][0] * 100,
        "draw": ensemble_probs[0][1] * 100,
        "away_win": ensemble_probs[0][2] * 100,
        "prediction": prediction
    }

st.set_page_config(
    page_title="FIFA Match Predictor",
    page_icon="⚽",
    layout="wide"
)

st.markdown("""
<style>

.stApp{
background:
linear-gradient(rgba(0,0,0,.75),rgba(0,0,0,.85)),
url('https://images.unsplash.com/photo-1574629810360-7efbbe195018?auto=format&fit=crop&w=2000&q=80');
background-size:cover;
background-position:center;
background-attachment:fixed;
}

.main-title{
text-align:center;
font-size:70px;
font-weight:900;
color:white;
}

.subtitle{
text-align:center;
font-size:20px;
color:#d1d5db;
margin-bottom:30px;
}

.match-card{
background:rgba(255,255,255,.08);
backdrop-filter:blur(15px);
border-radius:25px;
padding:40px;
text-align:center;
margin-top:20px;
margin-bottom:20px;
}

.result-card{
background:rgba(255,255,255,.08);
backdrop-filter:blur(15px);
border-radius:20px;
padding:25px;
text-align:center;
}

.winner-card{
background:linear-gradient(135deg,#f59e0b,#d97706);
border-radius:20px;
padding:30px;
text-align:center;
font-size:40px;
font-weight:900;
color:white;
}

</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="main-title">⚽ FIFA MATCH OUTCOME PREDICTOR</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Ensemble AI Prediction System</div>',
    unsafe_allow_html=True
)

teams = sorted(
    list(
        set(df["home_team"]).union(
            set(df["away_team"])
        )
    )
)

tournaments = sorted(
    df["tournament_group"]
    .dropna()
    .unique()
)

c1, c2 = st.columns(2)

with c1:
    home_team = st.selectbox(
        "Home Team",
        teams
    )

with c2:
    away_team = st.selectbox(
        "Away Team",
        teams,
        index=1
    )

tournament = st.selectbox(
    "Tournament",
    tournaments
)

neutral = st.checkbox(
    "Neutral Venue"
)

st.markdown(
    f"""
    <div class="match-card">
    <h1 style="font-size:65px;color:white;">
    {home_team.upper()}
    <br><br>
    VS
    <br><br>
    {away_team.upper()}
    </h1>
    </div>
    """,
    unsafe_allow_html=True
)

if st.button("🚀 PREDICT MATCH"):

    result = predict_match(
        home_team,
        away_team,
        tournament,
        neutral
    )

    r1, r2, r3 = st.columns(3)

    with r1:
        st.metric(
            f"{home_team} Win",
            f"{result['home_win']:.2f}%"
        )

    with r2:
        st.metric(
            "Draw",
            f"{result['draw']:.2f}%"
        )

    with r3:
        st.metric(
            f"{away_team} Win",
            f"{result['away_win']:.2f}%"
        )

    if result["prediction"] == 0:
        winner = home_team
    elif result["prediction"] == 1:
        winner = "DRAW"
    else:
        winner = away_team

    st.markdown(
        f"""
        <div class="winner-card">
        🏆 {winner.upper()}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.balloons()
