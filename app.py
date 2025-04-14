# app.py
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import json
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBRegressor, XGBClassifier
import pickle
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Path to your model and data
DATA_PATH = "/Users/tansintaj/F1ui/final_data_clean.csv"

# Load and process the dataset
def load_data():
    df = pd.read_csv(DATA_PATH)
    df.drop(columns=["Unnamed: 0", "Position"], inplace=True)
    
    # Store mappings for lookup
    event_teams = df[['EventName', 'Team']].drop_duplicates()
    event_drivers = df[['EventName', 'Team', 'Driver']].drop_duplicates()
    event_conditions = df[['EventName', 'eventYear', 'meanAirTemp', 'meanTrackTemp', 'Rainfall']].drop_duplicates()
    circuit_info = df[['EventName', 'CircuitLength']].drop_duplicates()
    
    # Encode categorical variables
    categorical_columns = ["EventName", "Team", "Compound", "Driver", "bestLapTimeIsFrom"]
    label_encoders = {}
    for col in categorical_columns:
        label_encoders[col] = LabelEncoder()
        df[col] = label_encoders[col].fit_transform(df[col])
    
    # Feature Engineering
    df["tyreDegradationPerStint"] = (df["deg_slope"] * df["StintLen"] + df["deg_bias"]) / df["StintLen"]
    df["fuelConsumptionPerStint"] = df["fuel_slope"] * df["StintLen"] + df["fuel_bias"]
    df["stintPerformance"] = df["StintLen"] / df["CircuitLength"]
    df["trackConditionIndex"] = (df["meanAirTemp"] + df["meanTrackTemp"]) - df["meanHumid"]
    
    # Targets for prediction
    df["totalPitStops"] = (df["designedLaps"] / df["StintLen"]).apply(np.ceil)
    df["nextPitLap"] = df["lapNumberAtBeginingOfStint"] + (df["StintLen"] // 2)
    df["nextTireCompound"] = df["Compound"]
    
    # Feature Selection
    selected_features = [
        "lapNumberAtBeginingOfStint", "eventYear", "meanHumid", "trackConditionIndex", "Rainfall",
        "designedLaps", "meanTrackTemp", "fuelConsumptionPerStint", "lag_slope_mean", "bestPreRaceTime",
        "CircuitLength", "StintLen", "RoundNumber", "stintPerformance", "tyreDegradationPerStint", "meanAirTemp"
    ]
    X = df[selected_features]
    y_pitstops = df["totalPitStops"]
    y_pitlap = df["nextPitLap"]
    y_tire = df["nextTireCompound"]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train-test split (not needed for final deployment)
    
    # Train the models
    pitstops_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    pitstops_model.fit(X_scaled, y_pitstops)
    
    pitlap_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    pitlap_model.fit(X_scaled, y_pitlap)
    
    tire_model = XGBClassifier(n_estimators=100, random_state=42)
    tire_model.fit(X_scaled, y_tire)
    
    # Store feature means for fallback
    feature_means = X.mean().to_dict()
    
    # Save models and related data for prediction
    model_data = {
        'pitstops_model': pitstops_model,
        'pitlap_model': pitlap_model,
        'tire_model': tire_model,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'feature_means': feature_means,
        'selected_features': selected_features,
        'df': df,
        'circuit_info': circuit_info,
        'event_teams': event_teams,
        'event_drivers': event_drivers,
        'event_conditions': event_conditions
    }
    
    with open('models.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    return model_data

# Check if models exist, if not, train them
if not os.path.exists('models.pkl'):
    print("Training models...")
    model_data = load_data()
else:
    print("Loading existing models...")
    with open('models.pkl', 'rb') as f:
        model_data = pickle.load(f)

# Function to get average values for a specific event/team/driver
def get_event_data(event_name, year=None, team=None, driver=None):
    df = model_data['df']
    selected_features = model_data['selected_features']
    feature_means = model_data['feature_means']
    label_encoders = model_data['label_encoders']
    
    # Filter the dataset based on available inputs
    filtered_df = df.copy()
    
    if event_name:
        # Get the encoded event name
        try:
            encoded_event = label_encoders["EventName"].transform([event_name])[0]
            filtered_df = filtered_df[filtered_df["EventName"] == encoded_event]
        except:
            # If event name is not found, use the original dataframe
            pass
    
    if year:
        filtered_df = filtered_df[filtered_df["eventYear"] == year]
        
    if team:
        try:
            encoded_team = label_encoders["Team"].transform([team])[0]
            filtered_df = filtered_df[filtered_df["Team"] == encoded_team]
        except:
            pass
    
    if driver:
        try:
            encoded_driver = label_encoders["Driver"].transform([driver])[0]
            filtered_df = filtered_df[filtered_df["Driver"] == encoded_driver]
        except:
            pass
    
    # If no data is available with the filters, use the overall dataset
    if filtered_df.empty:
        return {feature: feature_means[feature] for feature in selected_features}
    
    # Return average values for the selected features
    return {feature: filtered_df[feature].mean() if feature in filtered_df else feature_means[feature] 
            for feature in selected_features}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.json
        
        # Extract values
        track = data['track']
        year = data['year']
        team = data['team']
        driver = data['driver']
        air_temp = data['airTemp']
        track_temp = data['trackTemp']
        rainfall = data['rainfall']
        current_lap = data['currentLap']
        total_laps = data['totalLaps']
        stint_length = data['stintLength']
        
        # Get default values based on track, team, driver
        feature_values = get_event_data(track, year, team, driver)
        
        # Override with weather-related values
        feature_values["meanAirTemp"] = air_temp
        feature_values["meanTrackTemp"] = track_temp
        feature_values["Rainfall"] = rainfall
        feature_values["trackConditionIndex"] = (air_temp + track_temp) - feature_values["meanHumid"]
        
        # Set required user inputs
        feature_values["lapNumberAtBeginingOfStint"] = current_lap
        feature_values["StintLen"] = stint_length
        feature_values["designedLaps"] = total_laps
        
        # Get circuit length if available, otherwise use default
        circuit_info = model_data['circuit_info']
        circuit_rows = circuit_info[circuit_info['EventName'] == track]
        if not circuit_rows.empty:
            feature_values["CircuitLength"] = circuit_rows.iloc[0]['CircuitLength']
        
        # Update calculated features
        feature_values["stintPerformance"] = stint_length / feature_values["CircuitLength"]
        
        # Create feature vector in correct order
        selected_features = model_data['selected_features']
        new_input = np.array([feature_values[feature] for feature in selected_features]).reshape(1, -1)
        
        # Scale the input
        scaler = model_data['scaler']
        new_data_scaled = scaler.transform(new_input)
        
        # Make predictions
        pitstops_model = model_data['pitstops_model']
        pitlap_model = model_data['pitlap_model']
        tire_model = model_data['tire_model']
        
        pitstops = int(round(pitstops_model.predict(new_data_scaled)[0]))
        
        # Initialize lists for pit stops and tire compounds
        pit_stop_laps = []
        tire_compounds = []
        
        # Create a DataFrame for updating predictions
        new_df = pd.DataFrame(new_input, columns=selected_features)
        
        # Predict pit stops and tire compounds
        for _ in range(max(1, pitstops)):
            pitlap = int(round(pitlap_model.predict(new_data_scaled)[0]))
            next_tire_code = tire_model.predict(new_data_scaled)[0]
            
            # Convert tire code back to compound name
            label_encoders = model_data['label_encoders']
            next_tire = label_encoders["Compound"].inverse_transform([next_tire_code])[0]
            
            # Add to lists
            pit_stop_laps.append(pitlap)
            tire_compounds.append(next_tire)
            
            # Update lap number for the next stint
            current_lap = pitlap + 1
            new_df["lapNumberAtBeginingOfStint"] = current_lap
            new_data_scaled = scaler.transform(new_df)
        
        # Return results
        return jsonify({
            'pitstops': pitstops,
            'pit_lap_sequence': pit_stop_laps,
            'tire_sequence': tire_compounds
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)