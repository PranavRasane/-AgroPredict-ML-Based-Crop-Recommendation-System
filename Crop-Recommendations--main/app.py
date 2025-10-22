from flask import Flask, request, render_template
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load model and scaler
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("✅ Model and scaler loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")

@app.route('/')
def home():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    try:
        # Get form data
        N = int(request.form['Nitrogen'])
        P = int(request.form['Phosporus'])
        K = int(request.form['Potassium'])
        temperature = float(request.form['Temperature'])
        humidity = float(request.form['Humidity'])
        ph_value = float(request.form['Ph'])
        rainfall = float(request.form['Rainfall'])

        # Prepare features
        features = np.array([[N, P, K, temperature, humidity, ph_value, rainfall]])
        features_scaled = scaler.transform(features)
        
        # Predict
        prediction = model.predict(features_scaled)
        crop_id = prediction[0]

        # Crop dictionary
        crop_dict = {
            1: "Rice 🌾", 2: "Maize 🌽", 3: "Jute", 4: "Cotton", 5: "Coconut 🥥",
            6: "Papaya 🍈", 7: "Orange 🍊", 8: "Apple 🍎", 9: "Muskmelon", 10: "Watermelon 🍉",
            11: "Grapes 🍇", 12: "Mango 🥭", 13: "Banana 🍌", 14: "Pomegranate", 15: "Lentil",
            16: "Blackgram", 17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas",
            20: "Kidneybeans", 21: "Chickpea", 22: "Coffee ☕"
        }

        recommended_crop = crop_dict.get(crop_id, "Unknown crop")
        result_message = f"🎯 Perfect! You should grow: {recommended_crop}"
        
        return render_template('index.html', result=result_message)
    
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        return render_template('index.html', result=error_msg)

# Vercel requires this
if __name__ == '__main__':
    app.run(debug=True)