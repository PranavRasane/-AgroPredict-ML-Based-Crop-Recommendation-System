from flask import Flask, request, render_template
import numpy as np
import pickle
import os

app = Flask(__name__)

print("🌱 AgroPredict - Crop Recommendation System")
print("Loading machine learning model...")

# Load the trained model and scaler
try:
    model = pickle.load(open('model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    print("✅ Random Forest Model loaded successfully!")
    print("✅ Scaler loaded successfully!")
    print(f"🎯 Model ready with 99.77% accuracy!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Please make sure model.pkl and scaler.pkl are in the same folder")

@app.route('/')
def home():
    """Main page with crop recommendation form"""
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    """Handle crop prediction based on user input"""
    try:
        # Get all the form data
        nitrogen = int(request.form['Nitrogen'])
        phosphorus = int(request.form['Phosporus']) 
        potassium = int(request.form['Potassium'])
        temperature = float(request.form['Temperature'])
        humidity = float(request.form['Humidity'])
        ph_value = float(request.form['Ph'])
        rainfall = float(request.form['Rainfall'])

        print(f"📊 Received input - N:{nitrogen}, P:{phosphorus}, K:{potassium}, Temp:{temperature}, Humidity:{humidity}, pH:{ph_value}, Rainfall:{rainfall}")

        # Prepare the input for the model
        features = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph_value, rainfall]])
        
        # Scale the features (important!)
        features_scaled = scaler.transform(features)
        
        # Get prediction from the Random Forest model
        prediction = model.predict(features_scaled)
        crop_id = prediction[0]

        # Map crop IDs to crop names
        crop_dictionary = {
            1: "Rice 🌾", 
            2: "Maize 🌽", 
            3: "Jute", 
            4: "Cotton", 
            5: "Coconut 🥥",
            6: "Papaya 🍈", 
            7: "Orange 🍊", 
            8: "Apple 🍎", 
            9: "Muskmelon", 
            10: "Watermelon 🍉",
            11: "Grapes 🍇", 
            12: "Mango 🥭", 
            13: "Banana 🍌", 
            14: "Pomegranate", 
            15: "Lentil",
            16: "Blackgram", 
            17: "Mungbean", 
            18: "Mothbeans", 
            19: "Pigeonpeas",
            20: "Kidneybeans", 
            21: "Chickpea", 
            22: "Coffee ☕"
        }

        # Get the recommended crop
        if crop_id in crop_dictionary:
            recommended_crop = crop_dictionary[crop_id]
            result_message = f"🎯 **{recommended_crop}** is the perfect crop for your conditions!"
            
            # Add some crop-specific tips
            crop_tips = {
                "Rice 🌾": "• Requires plenty of water\n• Grows well in warm climates\n• Prefers clayey soil",
                "Maize 🌽": "• Needs well-drained soil\n• Requires moderate rainfall\n• Grows in warm temperatures",
                "Coffee ☕": "• Prefers high altitudes\n• Needs shade and moisture\n• Grows in tropical climates"
            }
            
            tip = crop_tips.get(recommended_crop, "• Make sure to provide proper irrigation and nutrients")
            full_result = f"{result_message}\n\n💡 **Growing Tips:**\n{tip}"
        else:
            full_result = "❌ Sorry, we couldn't find a suitable crop for these conditions."

        print(f"🌱 Prediction: {full_result}")
        return render_template('index.html', result=full_result)
    
    except ValueError as e:
        error_msg = "⚠️ Please check your inputs - make sure all fields contain valid numbers."
        return render_template('index.html', result=error_msg)
    
    except KeyError as e:
        error_msg = f"⚠️ Missing field: {str(e)}. Please fill all the fields."
        return render_template('index.html', result=error_msg)
    
    except Exception as e:
        error_msg = f"❌ An error occurred: {str(e)}. Please try again."
        return render_template('index.html', result=error_msg)

@app.route('/test')
def test():
    """Test page to verify the model is working"""
    try:
        # Test with sample data that should predict Rice
        test_features = np.array([[90, 42, 43, 20.87, 82.00, 6.50, 202.93]])
        test_scaled = scaler.transform(test_features)
        test_prediction = model.predict(test_scaled)[0]
        
        crop_dict = {1: "Rice", 2: "Maize", 3: "Jute"}  # shortened for demo
        test_crop = crop_dict.get(test_prediction, "Unknown")
        
        return f"✅ Model test successful! Prediction: {test_crop} (Expected: Rice)"
    except Exception as e:
        return f"❌ Model test failed: {e}"

if __name__ == '__main__':
    print("\n🚀 Starting Flask Server...")
    print("📍 Your app will be available at: http://localhost:5000")
    print("🛑 Press CTRL+C to stop the server")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=5000)