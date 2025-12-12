import pandas as pd
import numpy as np

# Set the number of rows for the dataset.
num_rows = 500
np.random.seed(42)  # For reproducibility

# ---------------------------
# Generate Input Columns
# ---------------------------
ages = np.random.randint(18, 70, size=num_rows)
genders = np.random.choice(["male", "female", "non-binary"], size=num_rows)
heights = np.round(np.random.uniform(150, 200, size=num_rows), 1)
weights = np.round(np.random.uniform(50, 100, size=num_rows), 1)
step_counts = np.random.randint(1000, 20000, size=num_rows)
distances = np.round(np.random.uniform(1, 15, size=num_rows), 2)
workout_types = np.random.choice(["running", "cycling", "yoga", "strength", "cardio"], size=num_rows)
workout_duration = np.random.randint(10, 121, size=num_rows)
heart_rate_max = np.random.randint(150, 201, size=num_rows)
heart_rate_resting = np.random.randint(50, 91, size=num_rows)
sleep_duration = np.round(np.random.uniform(4, 10, size=num_rows), 1)
sleep_quality = np.random.randint(1, 11, size=num_rows)
water_intake = np.round(np.random.uniform(0.5, 4.0, size=num_rows), 2)
bp_systolic_input = np.random.randint(90, 141, size=num_rows)
bp_diastolic_input = np.random.randint(60, 91, size=num_rows)
stress_levels = np.random.randint(1, 11, size=num_rows)
calories_consumed = np.random.randint(1500, 3501, size=num_rows)
protein_intake = np.round(np.random.uniform(30, 150, size=num_rows), 1)
carb_intake = np.round(np.random.uniform(100, 300, size=num_rows), 1)
fat_intake = np.round(np.random.uniform(20, 100, size=num_rows), 1)

# ---------------------------
# Generate Target Columns
# ---------------------------
calories_burned = np.random.randint(200, 801, size=num_rows)
fitness_level = np.random.choice(["beginner", "intermediate", "advanced"], size=num_rows)
heart_rate_avg = np.random.randint(60, 151, size=num_rows)
bmi = np.round(np.random.uniform(18, 35, size=num_rows), 1)
# For target blood pressure, generate similar ranges to the input ones.
bp_systolic_target = np.random.randint(90, 141, size=num_rows)
bp_diastolic_target = np.random.randint(60, 91, size=num_rows)

# ---------------------------
# Combine into a DataFrame
# ---------------------------
# Note: The training CSV contains both input and target columns.
data = {
    # Input Columns
    "age": ages,
    "gender": genders,
    "height_cm": heights,
    "weight_kg": weights,
    "step_count": step_counts,
    "distance_km": distances,
    "workout_type": workout_types,
    "workout_duration_min": workout_duration,
    "heart_rate_max": heart_rate_max,
    "heart_rate_resting": heart_rate_resting,
    "sleep_duration_hr": sleep_duration,
    "sleep_quality_score": sleep_quality,
    "water_intake_liters": water_intake,
    "blood_pressure_systolic": bp_systolic_input,
    "blood_pressure_diastolic": bp_diastolic_input,
    "stress_level": stress_levels,
    "calories_consumed": calories_consumed,
    "protein_intake_g": protein_intake,
    "carb_intake_g": carb_intake,
    "fat_intake_g": fat_intake,
    # Target Columns
    "calories_burned": calories_burned,
    "fitness level": fitness_level,
    "heart rate avg": heart_rate_avg,
    "bmi": bmi,
    # Note: The target blood pressure columns share the same names as input;
    # when using this CSV in your app, the code will pick the first occurrence for X and later for Y.
    "blood_pressure_systolic": bp_systolic_target,
    "blood_pressure_diastolic": bp_diastolic_target
}

df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv("sa.csv", index=False)

# Display the first few rows to verify
print(df.head())
