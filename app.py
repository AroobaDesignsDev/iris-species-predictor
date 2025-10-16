from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__, static_folder='static')


# Load the saved Decision Tree model
MODEL_PATH = "iris_decision_tree_model.pkl"
model = joblib.load(MODEL_PATH)

# Mapping species -> image filename (stored inside static/images/)
IMG_MAP = {
    "Iris-setosa": "iris-setosa.png",
    "Iris-versicolor": "iris-versicolor.png",
    "Iris-virginica": "iris-virginica.png"
}

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", prediction=None)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Read inputs
        sepal_length = float(request.form.get("sepal_length", 0))
        sepal_width  = float(request.form.get("sepal_width", 0))
        petal_length = float(request.form.get("petal_length", 0))
        petal_width  = float(request.form.get("petal_width", 0))

        arr = np.array([sepal_length, sepal_width, petal_length, petal_width]).reshape(1, -1)

        # Predict species
        pred = model.predict(arr)[0]

        # Probability (if supported by model)
        prob_info = None
        try:
            probs = model.predict_proba(arr)[0]
            classes = model.classes_.tolist()
            prob_info = list(zip(classes, [round(float(p), 4) for p in probs]))
        except Exception:
            pass

        # Directly assign image filename (no file existence check)
        img_filename = IMG_MAP.get(pred)

        return render_template(
            "index.html",
            prediction=pred,
            probabilities=prob_info,
            img_filename=img_filename,
            sepal_length=sepal_length,
            sepal_width=sepal_width,
            petal_length=petal_length,
            petal_width=petal_width
        )
    except Exception as e:
        return render_template("index.html", prediction=None, error=str(e))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8501, debug=True)
