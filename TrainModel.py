from tensorflow import keras
import pandas as pd
import sklearn.preprocessing as pp
import json
import joblib
import numpy as np
#import data
df = pd.read_csv('weather_data.csv')
#import dict
with open('numerization_dict.json', 'r') as f:
    numerization_dict = json.load(f)
#Numerize categorical data
df["Cloud Cover"] = df["Cloud Cover"].map(numerization_dict["Cloud Cover"])
df["Season"] = df["Season"].map(numerization_dict["Season"])
df["Location"] = df["Location"].map(numerization_dict["Location"])
#Separate features and labels
X = df.drop('Rain', axis=1)
y = df[['Rain']]
#Scale features 
scaler = pp.MinMaxScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, 'scaler_features.pkl')
#Build model
model = keras.Sequential([
    keras.layers.Dense(64,activation='swish',input_shape=(X.shape[1],)),
    keras.layers.Dense(32,activation='mish'),
    keras.layers.Dense(16,activation='swish'),
    keras.layers.Dense(8,activation='mish'),
    keras.layers.Dense(4,activation='swish'),
    keras.layers.Dense(2,activation='mish'),
    keras.layers.Dense(1,activation='sigmoid')
])
model.compile(optimizer='adamax', loss='binary_crossentropy', metrics=['accuracy'])
#Train model   
model.fit(X_scaled, y, epochs=150, batch_size=32, validation_split=0.2)
#Save model 
model.save('rain_prediction_model.h5')
#Evaluate model
loss, accuracy = model.evaluate(X_scaled, y)
print(f'Model Loss: {loss}, Model Accuracy: {accuracy}')