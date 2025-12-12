import streamlit as st
import base64
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Initialize session state variables
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "user_data" not in st.session_state:
    st.session_state.user_data = {}
if "show_signup" not in st.session_state:
    st.session_state.show_signup = False

# Define file paths and constants
DATA_FILE = 'data.csv'
USER_DATA_FILE = 'users.csv'

BACKGROUND_IMAGE = r'C:\Users\v santhosh kumar\Desktop\AICTE INTENSHIP\fitness.jpg'
LOGIN_BACKGROUND = r'C:\Users\v santhosh kumar\Desktop\AICTE INTENSHIP\io.jpg'

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

# Hide some inputs from the sidebar (they won't be shown to the user)
HIDDEN_INPUTS = [
    "sleep_quality_score", "fat_intake_g", "protein_intake_g",
    "calories_consumed", "carb_intake_g", "stress_level",
    "heart_rate_max", "heart_rate_resting"
]

def add_bg_from_local(image_file: str):
    """Set a fitness-themed animated background from a local image."""
    with open(image_file, "rb") as file:
        data = file.read()
    encoded = base64.b64encode(data).decode()
    st.markdown(
        f"""
        <style>
        @keyframes lightingEffect {{
            0% {{ filter: brightness(0.8); }}
            50% {{ filter: brightness(1.2); }}
            100% {{ filter: brightness(0.8); }}
        }}
        .stApp {{
            background: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
            animation: lightingEffect 6s infinite;
        }}
        .css-1d391kg, .css-18e3th9, .css-1kyxreq {{
            background-color: rgba(0, 0, 0, 0.5) !important;
            border-radius: 0.5rem;
        }}
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown p {{
            color: #FFFFFF;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def load_training_data():
    """Load training data and map fitness level from string to numeric."""
    df = pd.read_csv(DATA_FILE)
    df['fitness level'] = df['fitness level'].map({'beginner': 0, 'intermediate': 1, 'advanced': 2})
    return df

@st.cache_resource
def train_model():
    """Train a Random Forest model using the training data."""
    df = load_training_data()
    X = df[INPUT_COLS]
    X = pd.get_dummies(X, columns=["gender", "workout_type"], drop_first=True)
    Y = df[TARGET_COLS]
    X = X.astype(float)
    Y = Y.astype(float)
    xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.3, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(xtrain, ytrain)
    ypred = model.predict(xtest)
    r2 = r2_score(ytest, ypred)  # Computed but not displayed
    return model, r2, X.columns.tolist()

@st.cache_data
def get_feature_defaults():
    """Compute default values for each input column from training data."""
    df = load_training_data()
    defaults = {}
    for col in INPUT_COLS:
        if pd.api.types.is_numeric_dtype(df[col]):
            defaults[col] = df[col].mean()
        else:
            defaults[col] = df[col].mode()[0]
    return defaults

def preprocess_user_input(user_input, training_columns):
    """Convert user input into a DataFrame and one-hot encode as needed."""
    user_df = pd.DataFrame([user_input])
    defaults = get_feature_defaults()
    for col in [c for c in INPUT_COLS if c not in ["gender", "workout_type"]]:
        val = user_df.at[0, col]
        user_df.at[0, col] = float(val) if str(val).strip() != "" else defaults[col]
    for col in ["gender", "workout_type"]:
        if user_df.at[0, col] == "Select":
            user_df.at[0, col] = ""
    user_encoded = pd.get_dummies(user_df, columns=["gender", "workout_type"], drop_first=True)
    for col in training_columns:
        if col not in user_encoded.columns:
            user_encoded[col] = 0
    return user_encoded[training_columns].astype(float)

def preprocess_stored_data(df, training_columns):
    """Preprocess stored user data so it matches the training data format."""
    defaults = get_feature_defaults()
    for col in [c for c in INPUT_COLS if c not in ["gender", "workout_type"]]:
        df[col] = df[col].apply(lambda x: float(x) if pd.notnull(x) and str(x).strip() != "" else defaults[col])
    for col in ["gender", "workout_type"]:
        df[col] = df[col].apply(lambda x: "" if x == "Select" else x)
    df_encoded = pd.get_dummies(df, columns=["gender", "workout_type"], drop_first=True)
    for col in training_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    return df_encoded[training_columns].astype(float)

def load_user_data():
    """Load stored user data from CSV or create an empty DataFrame if missing."""
    if os.path.exists(USER_DATA_FILE):
        return pd.read_csv(USER_DATA_FILE)
    else:
        cols = ['username', 'password'] + INPUT_COLS
        return pd.DataFrame(columns=cols)

def save_user_data(df):
    """Save the user data DataFrame to CSV."""
    df.to_csv(USER_DATA_FILE, index=False)

def categorize_calories(calories):
    """
    Categorize the predicted calories burned into:
      - Fitness level: beginner, intermediate, advanced
      - heart rate avg: low, normal, high
      - Blood Pressure: low, normal, high
    Example thresholds (adjust as needed):
      calories < 250 => beginner, low, low
      250 <= calories < 500 => intermediate, normal, normal
      calories >= 500 => advanced, high, high
    """
    if calories < 250:
        return "beginner", "low", "low"
    elif calories < 500:
        return "intermediate", "normal", "normal"
    else:
        return "advanced", "high", "high"

def bmi_from_calories(calories):
    """
    Compute a BMI value based on predicted calories burned.
    (Hypothetical relationship.)
    """
    if calories < 250:
        return 30  
    elif calories < 500:
        return 25  
    else:
        return 22  

def add_bg_from_local_login():
    add_bg_from_local(LOGIN_BACKGROUND) 

def show_login():
    add_bg_from_local_login()
    st.markdown("""
        <style>
            .auth-title {
                text-align: center;
                font-size: 50px;
                color: #FFFFFF;
                margin-bottom: 30px;
            }
            .auth-header {
                text-align: center;
                font-size: 30px;
                color: #FFFFFF;
            }
            .stTextInput label, .stSelectbox label, .stSlider label {
                font-size: 20px;
            }
            .stButton button {
                font-size: 20px;
                padding: 8px 16px;
            }
        </style>
        """, unsafe_allow_html=True)
    st.markdown("<h1 class='auth-title'>User Authentication</h1>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h2 class='auth-header'>Login</h2>", unsafe_allow_html=True)
        username_input = st.text_input("Username", key="login_username")
        password_input = st.text_input("Password", type="password", key="login_password")
        
        btn_col1, btn_col2, _ = st.columns([1, 1, 5])
        with btn_col1:
            if st.button("Login", key="login_button"):
                users_df = load_user_data()
                if username_input in users_df['username'].values:
                    user_row = users_df[users_df['username'] == username_input].iloc[0]
                    if user_row['password'] == password_input:
                        st.session_state.logged_in = True
                        st.session_state.username = username_input
                        st.session_state.user_data = user_row[INPUT_COLS].to_dict()
                        st.success("Logged in successfully!")
                    else:
                        st.error("Incorrect password.")
                else:
                    st.error("Username not found.")
        with btn_col2:
            if st.button("Sign Up", key="signup_button"):
                st.session_state.show_signup = True
        if st.session_state.show_signup:
            st.markdown("<h2 class='auth-header'>Sign Up</h2>", unsafe_allow_html=True)
            signup_username = st.text_input("Choose a Username", key="signup_username")
            signup_password = st.text_input("Choose a Password", type="password", key="signup_password")
            if st.button("Register", key="register_button"):
                users_df = load_user_data()
                if signup_username in users_df['username'].values:
                    st.error("Username already exists. Please choose a different username.")
                else:
                    new_row = {"username": signup_username, "password": signup_password}
                    defaults = get_feature_defaults()
                    for col in INPUT_COLS:
                        new_row[col] = defaults[col]
                    users_df = pd.concat([users_df, pd.DataFrame([new_row])], ignore_index=True)
                    save_user_data(users_df)
                    st.success("Sign up successful! Please log in.")
                    st.session_state.show_signup = False

def show_main_app():
    add_bg_from_local(BACKGROUND_IMAGE)
    model, r2, training_columns = train_model()
    
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.user_data = {}
    
    st.sidebar.header("Input Features")
    
    df_loaded = load_training_data()
    user_input = {}
    
    for col in INPUT_COLS:
        if col in HIDDEN_INPUTS:
            defaults = get_feature_defaults()
            user_input[col] = st.session_state.user_data.get(col, defaults[col])
            continue
        
        if col in ["gender", "workout_type"]:
            if col == "gender":
                user_input[col] = st.sidebar.selectbox(
                    f"Input for {col}",
                    ["Select", "male", "female", "non-binary"],
                    index=0
                )
            else:
                workout_options = ["Select"] + sorted(list(df_loaded["workout_type"].dropna().unique()))
                user_input[col] = st.sidebar.selectbox(f"Input for {col}", workout_options, index=0)
        else:
            if col in ["age", "step_count"]:
                slider_min = int(df_loaded[col].min())
                slider_max = int(df_loaded[col].max())
                default_val = int(st.session_state.user_data.get(col, slider_min))
                user_input[col] = st.sidebar.slider(f"Input for {col}", slider_min, slider_max, default_val, step=1)
            elif col == "blood_pressure_diastolic":
                slider_min = 10.0
                slider_max = 140.0
                default_val = float(st.session_state.user_data.get(col, slider_min))
                user_input[col] = st.sidebar.slider(f"Input for {col}", slider_min, slider_max, default_val, step=1.0)
            elif col == "blood_pressure_systolic":
                slider_min = float(df_loaded[col].min())
                slider_max = 180.0
                default_val = float(st.session_state.user_data.get(col, slider_min))
                user_input[col] = st.sidebar.slider(f"Input for {col}", slider_min, slider_max, default_val, step=1.0)
            elif col == "height_cm":
                slider_min = float(df_loaded[col].min())
                slider_max = 250.0
                default_val = float(st.session_state.user_data.get(col, slider_min))
                user_input[col] = st.sidebar.slider(f"Input for {col}", slider_min, slider_max, default_val, step=0.1)
            elif col == "distance_km":
                slider_min = float(df_loaded[col].min())
                slider_max = 25.0
                default_val = float(st.session_state.user_data.get(col, slider_min))
                user_input[col] = st.sidebar.slider(f"Input for {col}", slider_min, slider_max, default_val, step=0.1)
            elif col == "workout_duration_min":
                slider_min = float(df_loaded[col].min()) / 60.0
                slider_max = 5.0
                if st.session_state.user_data.get(col) is not None:
                    default_val = float(st.session_state.user_data.get(col)) / 60.0
                else:
                    default_val = slider_min
                hours_value = st.sidebar.slider("Input for workout_duration (hours)", slider_min, slider_max, default_val, step=0.1)
                user_input[col] = hours_value * 60
            elif col == "water_intake_liters":
                slider_min = float(df_loaded[col].min())
                slider_max = 7.0
                default_val = float(st.session_state.user_data.get(col, slider_min))
                user_input[col] = st.sidebar.slider(f"Input for {col}", slider_min, slider_max, default_val, step=0.1)
            elif col == "sleep_duration_hr":
                slider_min = float(df_loaded[col].min())
                slider_max = 15.0
                default_val = float(st.session_state.user_data.get(col, slider_min))
                user_input[col] = st.sidebar.slider(f"Input for {col}", slider_min, slider_max, default_val, step=0.1)
            else:
                slider_min = float(df_loaded[col].min())
                slider_max = float(df_loaded[col].max())
                default_val = st.session_state.user_data.get(col, slider_min)
                user_input[col] = st.sidebar.slider(f"Input for {col}", slider_min, slider_max, float(default_val))
    
    st.markdown('<div class="css-1d391kg">', unsafe_allow_html=True)
    st.title("Fitness Prediction App")
    st.subheader("Empowering Your Fitness Journey")
    st.write("This application uses a Random Forest model to predict fitness outcomes. Update your data in the sidebar and click **Predict** to see the results.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    if st.sidebar.button("Predict"):
        st.markdown(
            """
            <style>
            .stTable table {
                background-color: #222222;
                color: white;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        
        user_encoded = preprocess_user_input(user_input, training_columns)
        prediction = model.predict(user_encoded)
        pred_df = pd.DataFrame(prediction, columns=TARGET_COLS)
        
        # Build a BP string
        pred_df["Blood Pressure"] = (
            pred_df["blood_pressure_systolic"].round().astype(int).astype(str) +
            "/" +
            pred_df["blood_pressure_diastolic"].round().astype(int).astype(str)
        )
        
        reverse_fitness_mapping = {0: 'beginner', 1: 'intermediate', 2: 'advanced'}
        pred_df["fitness level"] = pred_df["fitness level"].round().astype(int).map(reverse_fitness_mapping)
        
        # STEP 1: Get predicted calories and initial categorization
        calories_burned_pred = pred_df["calories_burned"].iloc[0]
        fit_cat, hr_cat, bp_cat = categorize_calories(calories_burned_pred)
        
        # STEP 2: Compute BMI based on calories burned
        bmi_value = bmi_from_calories(calories_burned_pred)
        
        # STEP 3: Override fitness level based on user input BP values if needed:
        # If user input blood_pressure_systolic > 120 and blood_pressure_diastolic > 60, then mark fitness level as "unfit".
        if (user_input["blood_pressure_systolic"] > 120) and (user_input["blood_pressure_diastolic"] > 60):
            fit_cat = "unfit"
        
        # Build final output DataFrame
        final_output = pd.DataFrame({
            "calories_burned": [calories_burned_pred],
            "fitness level": [fit_cat],
            "bmi": [bmi_value]
        })
        
        st.markdown('<div class="css-1d391kg">', unsafe_allow_html=True)
        st.subheader("Prediction Output")
        st.table(final_output)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Build recommendations based on predicted values and additional input parameters
        rec_messages = []
        if calories_burned_pred < 300:
            rec_messages.append(
                "Your predicted calorie burn is on the lower side. Consider increasing your workout intensity or duration to boost energy expenditure."
            )
        if bmi_value > 25:
            rec_messages.append(
                "Your computed BMI is above the ideal range. Evaluating your nutrition and incorporating regular cardio may help achieve a healthier balance."
            )
        if fit_cat == "beginner":
            rec_messages.append(
                "As your fitness level is predicted as 'beginner', focus on gradual progress with moderate exercises and build up strength over time."
            )
        if fit_cat == "unfit":
            rec_messages.append(
                "Your blood pressure inputs indicate high blood pressure; your fitness level is compromised. Please consult a healthcare professional."
            )
        if hr_cat == "low":
            rec_messages.append(
                "Your average heart rate is low, which may indicate excellent cardiovascular efficiency, but if you experience any symptoms, consult a professional."
            )
        elif hr_cat == "high":
            rec_messages.append(
                "Your average heart rate is high; consider reducing your workload or overexertion, and consult a healthcare provider if needed."
            )
        if bp_cat == "low":
            rec_messages.append("Your blood pressure is low; ensure you are well-hydrated and nourished.")
        elif bp_cat == "high":
            rec_messages.append("Your blood pressure is high; consider reducing stress and overexertion, and seek professional advice if necessary.")
        
        # Additional recommendations based on other input parameters:
        if user_input.get("water_intake_liters", 0) < 2.0:
            rec_messages.append("Your water intake seems low. Aim to drink at least 2 liters of water daily for optimal hydration.")
        if user_input.get("workout_duration_min", 0) < 30:
            rec_messages.append("Your workout duration appears short. Consider exercising for at least 30 minutes per session.")
        if user_input.get("step_count", 0) < 5000:
            rec_messages.append("Your daily step count is low. Increasing your steps can greatly improve your overall fitness.")
        if user_input.get("sleep_duration_hr", 0) < 7:
            rec_messages.append("You might not be getting enough sleep. Aim for at least 7 hours per night to help your body recover.")
        
        # Additional recommendations if input parameters are too high:
        if user_input.get("workout_duration_min", 0) > 120:
            rec_messages.append("Your workout duration appears very high. Consider reducing workout time to avoid overtraining.")
        if user_input.get("step_count", 0) > 20000:
            rec_messages.append("Your daily step count is extremely high; make sure to rest and recover adequately.")
        if user_input.get("water_intake_liters", 0) > 5:
            rec_messages.append("Your water intake is very high. Ensure you are not overhydrating unnecessarily.")
        if user_input.get("sleep_duration_hr", 0) > 10:
            rec_messages.append("Your sleep duration is high. While rest is important, ensure that oversleeping isn't affecting your daily routine.")
        
        if not rec_messages:
            rec_messages.append("Your predicted values look balanced. Keep maintaining your current fitness routine and adjust as needed.")
        
        recommendations_html = f"""
            <div style="background-color: #222222; padding: 15px; border-radius: 0.5rem; color: white;">
            <h3>Recommendations</h3>
            <ul>
                {''.join([f"<li>{msg}</li>" for msg in rec_messages])}
            </ul>
            </div>
        """
        st.markdown(recommendations_html, unsafe_allow_html=True)

# Set page config
st.set_page_config(page_title="Fitness Prediction App", layout="wide")

if not st.session_state.logged_in:
    show_login()
else:
    st.sidebar.success(f"Logged in as {st.session_state.username}")
    show_main_app()