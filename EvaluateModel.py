from tensorflow import keras
import pandas as pd
import sklearn.preprocessing as pp
import json
import joblib
import numpy as np
import sklearn.metrics as met
import matplotlib.pyplot as plt
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
y_test = df[['Rain']]
#Load scaler and scale features
scaler = joblib.load('scaler_features.pkl') 
X_scaled = scaler.transform(X)
#Load model
model = keras.models.load_model('rain_prediction_model.h5',compile=False)
#Predict
y_pred_prob = model.predict(X_scaled)
y_pred = (y_pred_prob > 0.5).astype(int)
#Evaluate model
print(met.confusion_matrix(y_test, y_pred))

precision = met.precision_score(y_test, y_pred)
recall = met.recall_score(y_test, y_pred)
f1 = met.f1_score(y_test, y_pred)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
#Plot ROC curve
fpr, tpr, thresholds = met.roc_curve(y_test, y_pred_prob)
roc_auc = met.auc(fpr, tpr)

plt.figure(figsize=(7, 7))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle='--')  # garis random guess

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()