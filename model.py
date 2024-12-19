# Cell 1: Import necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder

# Cell 2: Build the piracy detection model
def build_model():
    model = Sequential([
        Dense(64, activation='relu', input_shape=(2,)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Cell 3: Train the model with predefined data
def train_model():
    # Simulated dataset
    data = {
        'file_size_gb': [1.2, 0.8, 3.0, 2.5, 4.5, 1.1, 3.2, 0.9, 2.8, 5.0],
        'resolution': [720, 1080, 720, 1080, 1080, 720, 480, 720, 1080, 1080],
        'is_pirated': ['Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No']
    }

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(data['is_pirated'])  # Yes -> 1, No -> 0

    # Features
    X = np.array(list(zip(data['file_size_gb'], data['resolution'])))

    # Build and train model
    model = build_model()
    model.fit(X, y, epochs=20, batch_size=4, verbose=1)

    return model, label_encoder

# Cell 4: Predict piracy based on user input
def predict_piracy(model, label_encoder):
    try:
        file_size = float(input("Enter the file size in GB: "))
        resolution = int(input("Enter the resolution (e.g., 720, 1080): "))
        
        # Prepare data for prediction
        sample_data = np.array([[file_size, resolution]])
        prediction = model.predict(sample_data)

        # Decode the prediction
        predicted_label = (prediction > 0.5).astype(int)
        result = label_encoder.inverse_transform(predicted_label.ravel())[0]

        print(f"Piracy Detection Result: {result}")
    except Exception as e:
        print("Error in input or prediction:", e)

# Cell 5: Main execution
if __name__ == "__main__":
    print("Training the model...")
    model, label_encoder = train_model()
    print("Model training complete.")

    print("\nPiracy Detection System")
    predict_piracy(model, label_encoder)
