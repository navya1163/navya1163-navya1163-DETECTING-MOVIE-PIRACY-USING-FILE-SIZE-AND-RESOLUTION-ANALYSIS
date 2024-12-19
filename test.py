import numpy as np
from tensorflow.keras.models import load_model
import joblib

# Load model and label encoder
model = load_model('piracy_detection_model.h5')
label_encoder = joblib.load('label_encoder.pkl')

# Test data (adjust as needed)
test_data = np.array([[2.0, 1080]])  # Example: file_size=2.0GB, resolution=1080
prediction = model.predict(test_data)

# Decode prediction
predicted_label = (prediction > 0.5).astype(int)
result = label_encoder.inverse_transform(predicted_label.ravel())[0]

print(f"Test Input: {test_data}")
print(f"Prediction Probability: {prediction}")
print(f"Predicted Label: {result}")
