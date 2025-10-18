import pandas as pd
import numpy as np
import datetime

print("Script started. Defining data profiles...")

# Define the statistical "fingerprint" for each activity
# We use (mean, standard_deviation) for our random data
PROFILES = {
    "Normal_Resting": {
        "heart_rate": (75, 5),      # Calm heart rate
        "accel_x": (0, 0.1),        # Very little movement
        "accel_y": (0, 0.1),
        "accel_z": (1, 0.1),        # Z-axis gets 1g of gravity (phone is flat)
    },
    "Normal_Walking": {
        "heart_rate": (110, 10),    # Elevated heart rate
        "accel_x": (0, 0.5),        # Rhythmic, moderate shaking
        "accel_y": (1, 0.5),        # Simulating forward "push"
        "accel_z": (0, 0.5),
    },
    "Normal_Running": {
        "heart_rate": (150, 15),    # High heart rate
        "accel_x": (0, 1.5),        # High-impact, high variance
        "accel_y": (2, 1.5),        # Strong forward "push"
        "accel_z": (0, 1.5),
    },
    "Anomaly_Fall": {
        "heart_rate": (120, 10),    # Spiked from surprise/impact
        "accel_x": (8, 3),          # Extremely high, erratic sensor reading
        "accel_y": (8, 3),          # Represents the "impact"
        "accel_z": (8, 3),
    },
    "Anomaly_HealthEvent": {
        "heart_rate": (195, 5),     # Critically high (tachycardia)
        "accel_x": (0, 0.1),        # BUT... very little movement
        "accel_y": (0, 0.1),        # This is the key: high HR *while at rest*
        "accel_z": (1, 0.1),
    }
}

# This function creates a block of data for a given profile
def generate_data_block(label, duration_seconds, start_time):
    """Generates a DataFrame for a specific activity profile."""
    
    print(f"  Generating {duration_seconds}s of '{label}'...")
    
    num_samples = duration_seconds
    timestamps = pd.date_range(start=start_time, periods=num_samples, freq='S')
    
    # --- Handle special anomaly cases ---
    if label == "Anomaly_SensorFailure":
    # All sensors report a stuck value of -1
        data = {
            "timestamp": timestamps,
            "heart_rate": np.full(num_samples, -1),
            "accel_x": np.full(num_samples, -1),
            "accel_y": np.full(num_samples, -1),
            "accel_z": np.full(num_samples, -1),
            "label": label
        }
        return pd.DataFrame(data)

    if label == "Anomaly_DataLoss":
        # All sensors report "Not a Number" (NaN)
        data = {
            "timestamp": timestamps,
            "heart_rate": np.full(num_samples, np.nan),
            "accel_x": np.full(num_samples, np.nan),
            "accel_y": np.full(num_samples, np.nan),
            "accel_z": np.full(num_samples, np.nan),
            "label": label
        }
        return pd.DataFrame(data)

    # --- Handle standard profiles ---
    profile = PROFILES[label]
    
    data = {
        "timestamp": timestamps,
        "heart_rate": np.random.normal(profile["heart_rate"][0], profile["heart_rate"][1], num_samples),
        "accel_x": np.random.normal(profile["accel_x"][0], profile["accel_x"][1], num_samples),
        "accel_y": np.random.normal(profile["accel_y"][0], profile["accel_y"][1], num_samples),
        "accel_z": np.random.normal(profile["accel_z"][0], profile["accel_z"][1], num_samples),
        "label": label
    }
    
    return pd.DataFrame(data)

# --- Main script ---
if __name__ == "__main__":
    
    all_data_blocks = []
    current_time = datetime.datetime.now()
    
    # --- Generate NORMAL Data (the majority) ---
    # 60 minutes of Resting
    block = generate_data_block("Normal_Resting", 3600, current_time)
    all_data_blocks.append(block)
    current_time += datetime.timedelta(seconds=3600)
    
    # 60 minutes of Walking
    block = generate_data_block("Normal_Walking", 3600, current_time)
    all_data_blocks.append(block)
    current_time += datetime.timedelta(seconds=3600)
    
    # 30 minutes of Running
    block = generate_data_block("Normal_Running", 1800, current_time)
    all_data_blocks.append(block)
    current_time += datetime.timedelta(seconds=1800)
    
    # --- Generate ANOMALY Data (rare events) ---
    # 10 separate 10-second "Fall" events
    for _ in range(10):
        block = generate_data_block("Anomaly_Fall", 10, current_time)
        all_data_blocks.append(block)
        current_time += datetime.timedelta(seconds=10)

    # 5 separate 20-second "Health Event" events
    for _ in range(5):
        block = generate_data_block("Anomaly_HealthEvent", 20, current_time)
        all_data_blocks.append(block)
        current_time += datetime.timedelta(seconds=20)
        
    # 5 separate 10-second "Sensor Failure" events
    for _ in range(5):
        block = generate_data_block("Anomaly_SensorFailure", 10, current_time)
        all_data_blocks.append(block)
        current_time += datetime.timedelta(seconds=10)
        
    # 5 separate 10-second "Data Loss" events
    # (This is your "NoHeartRateAndAccelaration" label)
    for _ in range(5):
        block = generate_data_block("Anomaly_DataLoss", 10, current_time)
        all_data_blocks.append(block)
        current_time += datetime.timedelta(seconds=10)

    # --- Combine and Save ---
    print("\nCombining all data blocks...")
    final_df = pd.concat(all_data_blocks, ignore_index=True)
    
    # Shuffle the data so anomalies aren't all at the end
    final_df = final_df.sample(frac=1).reset_index(drop=True)
    
    output_path = "data/simulated_smartwatch_data.csv"
    final_df.to_csv(output_path, index=False)
    
    print(f"\n--- SCRIPT COMPLETE ---")
    print(f"Total samples generated: {len(final_df)}")
    print(f"Data saved to: {output_path}")