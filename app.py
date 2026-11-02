from flask import Flask, render_template, request, jsonify
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
app = Flask(__name__)

# Educational demo model trained on generated sample data (not official/live AQI data).
rng = np.random.default_rng(42)
n = 2500
pm25 = rng.uniform(0, 300, n)
pm10 = rng.uniform(0, 400, n)
no2 = rng.uniform(0, 150, n)
so2 = rng.uniform(0, 100, n)
co = rng.uniform(0, 15, n)
temp = rng.uniform(5, 45, n)
humidity = rng.uniform(10, 100, n)
X = np.column_stack([pm25, pm10, no2, so2, co, temp, humidity])
score = pm25*0.55 + pm10*0.18 + no2*0.25 + so2*0.12 + co*8 + rng.normal(0, 12, n)
y = np.where(score < 50, "Good", np.where(score < 100, "Moderate", np.where(score < 170, "Poor", np.where(score < 250, "Very Poor", "Severe"))))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
model = RandomForestClassifier(n_estimators=140, max_depth=12, random_state=42, class_weight="balanced")
model.fit(X_train, y_train)
MODEL_ACCURACY = accuracy_score(y_test, model.predict(X_test))

@app.route("/")
def home():
    return render_template("index.html", accuracy=round(MODEL_ACCURACY*100, 1))

@app.post("/predict")
def predict():
    data = request.get_json(force=True)
    features = [[float(data[k]) for k in ["pm25","pm10","no2","so2","co","temp","humidity"]]]
    pred = model.predict(features)[0]
    probs = model.predict_proba(features)[0]
    confidence = round(float(max(probs))*100, 1)
    return jsonify({"category": pred, "confidence": confidence, "model_accuracy": round(MODEL_ACCURACY*100,1),
                   "note":"Demo prediction from a Random Forest trained on synthetic educational data. Not official AQI."})

if __name__ == "__main__":
    app.run(debug=True)
