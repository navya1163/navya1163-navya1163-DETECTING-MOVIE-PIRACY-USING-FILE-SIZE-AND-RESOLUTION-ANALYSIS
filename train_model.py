import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import save_model

# Build the model
def build_model():
    model = Sequential([
        Dense(64, activation='relu', input_shape=(2,)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Training
def train_and_save_model():
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

    # Save the model and label encoder
    model.save('piracy_detection_model.h5')
    import joblib
    joblib.dump(label_encoder, 'label_encoder.pkl')

train_and_save_model()
