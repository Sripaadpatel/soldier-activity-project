"# Soldier Activity & Anomaly Detection" 
# Anomaly Detection and Activity Recognition for Soldier Welfare

**Project Status:** Completed

This project builds a machine learning system to monitor a soldier's well-being and tactical situation using data from a smartwatch. The final model can successfully classify 3 normal activities and 4 distinct anomalies, including critical health and fall events.

---

## 1. Project Goal

The system is designed to fulfill two primary functions:
* **Soldier Welfare:** Provide real-time alerts by detecting anomalous events like falls or health crises.
* **Tactical Awareness:** Provide a high-level overview of a soldier's current activity state (e.g., "Resting" or "Running").

---

## 2. The Data

The project uses a **simulated smartwatch dataset** (`data/simulated_smartwatch_data.csv`). This was necessary because public datasets (like UCI-HAR) do not contain anomaly data or heart rate features.

The simulated dataset contains 4 raw features and 7 specific labels:

### Features (The "Inputs"):
* `heart_rate`
* `accel_x` (X-axis acceleration)
* `accel_y` (Y-axis acceleration)
* `accel_z` (Z-axis acceleration)

### Labels (The "Outputs"):
1.  **Normal_Resting:** Low heart rate, low movement.
2.  **Normal_Walking:** Elevated heart rate, moderate rhythmic movement.
3.  **Normal_Running:** High heart rate, high-impact movement.
4.  **Anomaly_Fall:** High, erratic acceleration.
5.  **Anomaly_HealthEvent:** Critically high heart rate *without* movement.
6.  **Anomaly_SensorFailure:** All sensors report an error state (`-1`).
7.  **Anomaly_DataLoss:** All sensors report missing data (`0`).

---

## 3. The Model

The final model is a **Random Forest Classifier** trained on the 7 labels.

* **Model File:** `models/soldier_smartwatch_model.joblib`
* **Training Notebook:** `notebooks/03-Smartwatch-Model.ipynb`
* **Performance:** The model achieved **~99-100% accuracy** on the test set, successfully distinguishing all 7 normal and anomalous classes.

---

## 4. How to Use This Project

1.  **Environment:** The project requires the Conda environment defined in `soldier_env`.
2.  **Generate Data:** Run `python src/generate_smartwatch_data.py` to create the dataset.
3.  **Train Model:** Open and run the `notebooks/03-Smartwatch-Model.ipynb` notebook. This will train the model and save it to the `models/` folder.