from flask import Flask, request, render_template
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import webbrowser
from threading import Timer

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and label encoder
model = load_model('piracy_detection_model.h5')
label_encoder = joblib.load('label_encoder.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file_size = float(request.form['file_size'])
        resolution = int(request.form['resolution'])

        # Debug: Print the inputs
        print(f"Input file_size: {file_size}, resolution: {resolution}")

        # Prepare data for prediction
        sample_data = np.array([[file_size, resolution]])
        print(f"Formatted input for model: {sample_data}")

        prediction = model.predict(sample_data)
        print(f"Raw model prediction: {prediction}")

        # Decode the prediction
        predicted_label = (prediction > 0.5).astype(int)
        print(f"Thresholded prediction: {predicted_label}")

        result = label_encoder.inverse_transform(predicted_label.ravel())[0]
        print(f"Final result: {result}")

        return render_template('result.html', result=result)
    except Exception as e:
        return f"Error: {e}"


def open_browser():
    """Open the default web browser to the Flask app URL."""
    webbrowser.open_new("http://127.0.0.1:5000/")

if __name__ == "__main__":
    # Start a timer to open the browser after the app runs
    Timer(1, open_browser).start()
    app.run(debug=True)
