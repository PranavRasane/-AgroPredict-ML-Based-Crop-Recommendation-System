from flask import Flask, request, render_template
import numpy as np
import pickle

# Importing model
model = pickle.load(open('model.pkl', 'rb'))

# Creating Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    N = int(request.form['Nitrogen'])
    P = int(request.form['Phosporus'])
    K = int(request.form['Potassium'])
    temperature = float(request.form['Temperature'])
    humidity = float(request.form['Humidity'])
    ph = int(request.form['Ph'])
    rainfall = float(request.form['Rainfall'])

    features = [N, P, K, temperature, humidity, ph, rainfall]
    single_prediction = np.array(features).reshape(1, -1)
    prediction = model.predict(single_prediction)

    crop_dict = {
        1: "rice",
        2: "maize",
        3: "jute",
        4: "cotton",
        5: "coconut",
        6: "papaya",
        7: "orange",
        8: "apple",
        9: "muskmelon",
        10: "watermelon",
        11: "grapes",
        12: "mango",
        13: "banana",
        14: "pomegranate",
        15: "lentil",
        16: "blackgram",
        17: "mungbean",
        18: "mothbeans",
        19: "pigeonpeas",
        20: "kidneybeans",
        21: "chickpea",
        22: "coffee"
    }

    # Lookup the crop name based on prediction (prediction[0] is the crop ID)
    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result = "{} is the best crop to be cultivated.".format(crop)
    else:
        result = "Sorry, we are not able to recommend a proper crop for this environment."
    
    return render_template('index.html', result=result)

# Main function to run the app
if __name__ == "__main__":
    app.run(debug=True)
