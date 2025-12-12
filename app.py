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
# FILE PATH FIX
# -----------------------------
BASE_DIR = os.path.dirname(__file__)

DATA_FILE = os.path.join(BASE_DIR, "data.csv")
USER_DATA_FILE = os.path.join(BASE_DIR, "users.csv")

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
# BACKGROUND IMAGE FIX
# -----------------------------
def add_bg(image_path):
    if not os.path.exists(image_path):
        st.warning(f"⚠️ Missing background image: {image_path}")
        return

    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# -----------------------------
# DATA LOADING
# -----------------------------
def load_training_data():
    if not os.path.exists(DATA_FILE):
        st.error("❌ data.csv not found. Upload data.csv to your repository.")
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

def preprocess_user_input(data, training_cols):
    df = pd.DataFrame([data])
    defaults = get_feature_defaults()

    for col in INPUT_COLS:
        if col not in ["gender", "workout_type"]:
            df[col] = float(df[col]) if df[col].values[0] != "" else defaults[col]

    df = pd.get_dummies(df, columns=["gender", "workout_type"], drop_first=True)

    for col in training_cols:
        if col not in df.columns:
            df[col] = 0

    return df[training_cols].astype(float)

# -----------------------------
# USER SYSTEM (LOGIN/SIGNUP)
# -----------------------------
def load_user_data():
    if os.path.exists(USER_DATA_FILE):
        return pd.read_csv(USER_DATA_FILE)
    return pd.DataFrame(columns=["username", "password"] + INPUT_COLS)

def save_user_data(df):
    df.to_csv(USER_DATA_FILE, index=False)

# -----------------------------
# RECOMMENDATION ENGINE
# -----------------------------
def generate_recommendations(calories_burned, bmi_value, fit_cat, hr_cat, bp_cat, user_input):
    rec = []

    if calories_burned < 300:
        rec.append("Your calorie burn is low. Try increasing workout intensity.")

    if bmi_value > 25:
        rec.append("BMI is high. Include more cardio and reduce calorie intake.")

    if fit_cat == "unfit":
        rec.append("High BP detected. Consult a doctor before heavy exercise.")

    if hr_cat == "high":
        rec.append("Heart rate is high. Reduce exertion and take breaks.")

    if user_input.get("water_intake_liters", 0) < 2:
        rec.append("Increase water intake to at least 2 liters per day.")

    if user_input.get("sleep_duration_hr", 0) < 7:
        rec.append("You need more sleep. Aim for at least 7 hours.")

    if not rec:
        rec.append("Great performance! Keep your routine consistent.")

    return rec

# -----------------------------
# LOGIN PAGE
# -----------------------------
def show_login():
    add_bg(LOGIN_BACKGROUND)

    st.title("🔐 Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        df = load_user_data()
        if username in df["username"].values:
            row = df[df["username"] == username].iloc[0]
            if row["password"] == password:
                st.success("Login success!")
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.user_data = row[INPUT_COLS].to_dict()
            else:
                st.error("Invalid password")
        else:
            st.error("User not found")

    if st.button("Sign Up"):
        st.session_state.show_signup = True

    if st.session_state.show_signup:
        st.subheader("Create account")
        new_user = st.text_input("New username")
        new_pass = st.text_input("New password", type="password")

        if st.button("Register"):
            df = load_user_data()
            if new_user in df["username"].values:
                st.error("Username exists!")
            else:
                defaults = get_feature_defaults()
                row = {"username": new_user, "password": new_pass}
                row.update(defaults)
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                save_user_data(df)
                st.success("Account created!")

# -----------------------------
# MAIN APP
# -----------------------------
def show_main_app():
    add_bg(BACKGROUND_IMAGE)

    st.sidebar.title("User Panel")
    model, r2, training_cols = train_model()

    df_data = load_training_data()
    defaults = get_feature_defaults()
    user_input = {}

    st.title("🏋 Fitness Prediction App")

    for col in INPUT_COLS:
        if col in HIDDEN_INPUTS:
            user_input[col] = defaults[col]
            continue

        if col == "gender":
            user_input[col] = st.sidebar.selectbox(col, ["male", "female", "non-binary"])

        elif col == "workout_type":
            options = sorted(df_data["workout_type"].unique())
            user_input[col] = st.sidebar.selectbox(col, options)

        else:
            user_input[col] = st.sidebar.number_input(col, value=float(defaults[col]))

    if st.sidebar.button("Predict"):
        X = preprocess_user_input(user_input, training_cols)
        pred = model.predict(X)[0]

        pred_df = pd.DataFrame([pred], columns=TARGET_COLS)

        calories = pred_df["calories_burned"].iloc[0]
        bmi_val = pred_df["bmi"].iloc[0]
        fit_cat = "beginner"  # simple placeholder
        hr_cat = "normal"
        bp_cat = "normal"

        st.subheader("📊 Prediction Result")
        st.table(pred_df)

        rec = generate_recommendations(calories, bmi_val, fit_cat, hr_cat, bp_cat, user_input)

        st.subheader("💡 Recommendations")
        for r in rec:
            st.write("✔", r)

# -----------------------------
# PAGE ROUTER
# -----------------------------
st.set_page_config(page_title="Fitness Tracker", layout="wide")

if not st.session_state.logged_in:
    show_login()
else:
    show_main_app()
