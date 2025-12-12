import streamlit as st
import base64
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# -----------------------------
# SESSION STATE
# -----------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "user_data" not in st.session_state:
    st.session_state.user_data = {}
if "show_signup" not in st.session_state:
    st.session_state.show_signup = False

# -----------------------------
# FILE PATH FIX (IMPORTANT)
# -----------------------------
BASE_DIR = os.path.dirname(__file__)
DATA_FILE = os.path.join(BASE_DIR, "data.csv")
USER_DATA_FILE = os.path.join(BASE_DIR, "users.csv")

# IMAGE PATHS INSIDE REPO
BACKGROUND_IMAGE = os.path.join(BASE_DIR, "assets", "fitness.jpg")
LOGIN_BACKGROUND = os.path.join(BASE_DIR, "assets", "login.jpg")

# -----------------------------
# COLUMN DEFINITIONS
# -----------------------------
INPUT_COLS = [
    "age", "gender", "height_cm", "weight_kg", "step_count", "distance_km",
    "workout_type", "workout_duration_min", "heart_rate_max", "heart_rate_resting",
    "sleep_duration_hr", "sleep_quality_score", "water_intake_liters",
    "blood_pressure_systolic", "blood_pressure_diastolic", "stress_level",
    "calories_consumed", "protein_intake_g", "carb_intake_g", "fat_intake_g"
]

TARGET_COLS = [
    "calories_burned", "fitness level", "heart rate avg", "bmi",
    "blood_pressure_systolic", "blood_pressure_diastolic"
]

HIDDEN_INPUTS = [
    "sleep_quality_score", "fat_intake_g", "protein_intake_g",
    "calories_consumed", "carb_intake_g", "stress_level",
    "heart_rate_max", "heart_rate_resting"
]

# -----------------------------
# BACKGROUND IMAGE LOADER FIX
# -----------------------------
def add_bg_from_local(image_file):
    if not os.path.exists(image_file):
        st.warning(f"⚠️ Background image file missing: {image_file}")
        return

    with open(image_file, "rb") as file:
        encoded = base64.b64encode(file.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def add_bg_from_local_login():
    add_bg_from_local(LOGIN_BACKGROUND)

# -----------------------------
# DATA LOADING
# -----------------------------
def load_training_data():
    if not os.path.exists(DATA_FILE):
        st.error("❌ data.csv not found in the repository!")
        return pd.DataFrame()

    df = pd.read_csv(DATA_FILE)
    df["fitness level"] = df["fitness level"].map(
        {"beginner": 0, "intermediate": 1, "advanced": 2}
    )
    return df

@st.cache_resource
def train_model():
    df = load_training_data()
    X = pd.get_dummies(df[INPUT_COLS], columns=["gender", "workout_type"], drop_first=True)
    Y = df[TARGET_COLS].astype(float)

    X = X.astype(float)

    xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.3, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(xtrain, ytrain)

    r2 = r2_score(ytest, model.predict(xtest))
    return model, r2, X.columns.tolist()

@st.cache_data
def get_feature_defaults():
    df = load_training_data()
    defaults = {}

    for col in INPUT_COLS:
        if col not in df.columns:
            defaults[col] = 0
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            defaults[col] = df[col].mean()
        else:
            defaults[col] = df[col].mode()[0]

    return defaults

def preprocess_user_input(user_input, training_columns):
    df = pd.DataFrame([user_input])
    defaults = get_feature_defaults()

    for col in INPUT_COLS:
        if col not in ["gender", "workout_type"]:
            df[col] = float(df[col]) if df[col].values[0] != "" else defaults[col]

    df = pd.get_dummies(df, columns=["gender", "workout_type"], drop_first=True)

    for col in training_columns:
        if col not in df.columns:
            df[col] = 0

    return df[training_columns].astype(float)

# -----------------------------
# USER DATA FUNCTIONS
# -----------------------------
def load_user_data():
    if os.path.exists(USER_DATA_FILE):
        return pd.read_csv(USER_DATA_FILE)
    return pd.DataFrame(columns=["username", "password"] + INPUT_COLS)

def save_user_data(df):
    df.to_csv(USER_DATA_FILE, index=False)

# -----------------------------
# LOGIN SCREEN
# -----------------------------
def show_login():
    add_bg_from_local_login()
    st.title("🔐 Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        df = load_user_data()
        if username in df["username"].values:
            row = df[df["username"] == username].iloc[0]
            if row["password"] == password:
                st.success("Logged in successfully!")
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.user_data = row[INPUT_COLS].to_dict()
            else:
                st.error("Wrong password")
        else:
            st.error("User not found")

    if st.button("Sign Up"):
        st.session_state.show_signup = True

    if st.session_state.show_signup:
        st.subheader("Create Account")
        new_user = st.text_input("New Username")
        new_pass = st.text_input("New Password", type="password")

        if st.button("Register"):
            df = load_user_data()
            if new_user in df["username"].values:
                st.error("Username already exists")
            else:
                defaults = get_feature_defaults()
                new_row = {"username": new_user, "password": new_pass}
                new_row.update(defaults)
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                save_user_data(df)
                st.success("Account created! Please log in.")
                st.session_state.show_signup = False

# -----------------------------
# MAIN APP
# -----------------------------
def show_main_app():
    add_bg_from_local(BACKGROUND_IMAGE)

    st.sidebar.title(f"Welcome {st.session_state.username}")
    model, r2, columns = train_model()

    df = load_training_data()
    defaults = get_feature_defaults()
    user_input = {}

    for col in INPUT_COLS:
        if col in HIDDEN_INPUTS:
            user_input[col] = defaults[col]
            continue

        if col == "gender":
            user_input[col] = st.sidebar.selectbox(col, ["male", "female", "non-binary"])

        elif col == "workout_type":
            options = sorted(df["workout_type"].unique())
            user_input[col] = st.sidebar.selectbox(col, options)

        else:
            minv = float(df[col].min())
            maxv = float(df[col].max())
            user_input[col] = st.sidebar.slider(col, minv, maxv, defaults[col])

    st.title("🏋 Fitness Prediction App")

    if st.sidebar.button("Predict"):
        X = preprocess_user_input(user_input, columns)
        pred = model.predict(X)[0]

        result = pd.DataFrame([pred], columns=TARGET_COLS)
        st.subheader("📊 Prediction Output")
        st.table(result)

# -----------------------------
# PAGE ROUTING
# -----------------------------
st.set_page_config(page_title="Fitness App", layout="wide")

if not st.session_state.logged_in:
    show_login()
else:
    show_main_app()
