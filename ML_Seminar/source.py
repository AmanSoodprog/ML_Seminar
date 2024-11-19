import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
import pandas as pd

# Load the dataset
df = pd.read_csv('synthetic_book_dataset.csv')

# Preprocess the data
title_encoder = LabelEncoder()
genre_encoder = LabelEncoder()
description_encoder = OneHotEncoder(sparse_output=False)

# Encode the titles and genres
df["Title_Encoded"] = title_encoder.fit_transform(df["Title"])
df["Genre_Encoded"] = genre_encoder.fit_transform(df["Genre"])

# Encode the descriptions using OneHotEncoder
description_encoded = description_encoder.fit_transform(df["Description"].values.reshape(-1, 1))

# Concatenate the one-hot encoded descriptions with the original DataFrame
df_encoded = pd.DataFrame(description_encoded, columns=description_encoder.categories_[0])
df = pd.concat([df, df_encoded], axis=1)

# Prepare input features (Title_Encoded, Genre_Encoded) and target (one-hot encoded description)
X = np.array(df[["Title_Encoded", "Genre_Encoded"]])
y = np.array(df[df_encoded.columns])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model definition
def build_model():
    model = Sequential([
        Dense(128, activation='relu', input_shape=(2,)),
        Dropout(0.3),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(y_train.shape[1], activation='softmax')  # Output layer with softmax for multi-class classification
    ])
    return model

learning_rates = [0.0001, 0.01, 0.5] 

models = {}
for lr in learning_rates:
    optimizer = Adam(learning_rate=lr)
    model = build_model()
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    models[lr] = model

history = {}
for lr, model in models.items():
    print(f"\nTraining model with learning rate: {lr}\n")
    history[lr] = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                            epochs=10, batch_size=32, verbose=1)

for lr, hist in history.items():
    train_acc = hist.history['accuracy']
    val_acc = hist.history['val_accuracy']
    print(f"\nLearning Rate: {lr}")
    print(f"Final Training Accuracy: {train_acc[-1]:.4f}")
    print(f"Final Validation Accuracy: {val_acc[-1]:.4f}")


sample_input = np.array([[title_encoder.transform(["Journey to Mars"])[0],
                          genre_encoder.transform(["Science Fiction"])[0]]])

print("\nSample Input:")
print("Title: Journey to Mars, Genre: Science Fiction")

for lr, model in models.items():
    prediction = model.predict(sample_input)
    decoded_description = description_encoder.inverse_transform(prediction)
    print(f"\nLearning Rate: {lr}")
    print(f"Generated Description: {decoded_description[0]}")
