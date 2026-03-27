from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from ml_model import WeatherPredictor

app = Flask(__name__)
# Enable CORS so users can test standalone index.html files outside of the Flask server
CORS(app)

# Global model instance
model = WeatherPredictor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    country = data.get('country', 'India')
    
    # Load data for the country and trigger prediction
    success = model.load_data(country)
    if not success:
        return jsonify({"error": "Failed to load data for the specified country."}), 400
        
    prediction_results = model.train_and_predict()
    if "error" in prediction_results:
        return jsonify(prediction_results), 400
        
    return jsonify(prediction_results)

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    aeroData = data.get('aeroData', {})
    
    # Generate the dynamic AERO response
    loc = aeroData.get('location', 'Unknown')
    time = aeroData.get('time', 'Future')
    type_ = aeroData.get('type', 'Weather')
    details = aeroData.get('detail', 'Basic')
    units = aeroData.get('units', 'Metric')
    usecase = aeroData.get('usecase', 'General')
    risk = aeroData.get('risk', 'Low Risk')
    
    # Build dynamic response imitating streaming output
    response = f">> AERO QUANTUM SCAN COMPLETE <<\n"
    response += f"Target Coordinates: {loc}\n"
    response += f"Temporal Window: {time}\n\n"
    
    if units == "Metric":
        temp = "34°C"
        wind = "120 km/h"
    else:
        temp = "93°F"
        wind = "75 mph"
        
    if usecase == "Farmer":
        response += f"Agricultural Insight: Sub-soil moisture levels expected to drop. Prepare irrigation.\n"
    elif usecase == "Aviation":
        response += f"Aviation Advisory: High altitude turbulence predicted due to anti-gravity flux.\n"
        
    if risk == "High Risk":
        response += f"CRITICAL ALERT: Class 4 Atmospheric anomalies detected. Severe environmental disruption imminent.\n"
        
    response += f"General Analysis: The {type_} patterns over {loc} show significant fluctuations. "
    response += f"Temperature standard deviations point to a baseline of {temp} with wind currents of {wind}. "
    
    if details == "Advanced":
        response += f"\n[Advanced Telemetrics]: Barometric pressure at 940hPa. Zero-point energy field stable at 0.04% deviance."
        
    response += "\n\nWould you like to analyze another temporal window? (Yes/No)"
    
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
