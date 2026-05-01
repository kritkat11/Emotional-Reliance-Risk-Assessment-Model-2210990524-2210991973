import os
from flask import Flask, jsonify, render_template, request

from model.train_model import predict_risk

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}
    text = str(payload.get("text", "")).strip()

    if not text:
        return jsonify({"error": "Text input is required."}), 400

    try:
        result = predict_risk(text)
        return jsonify(result), 200
    except Exception:
        return jsonify({"error": "Prediction failed. Please try again."}), 500

if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "5055"))
    debug = os.getenv("FLASK_DEBUG", "1") == "1"
    app.run(host=host, port=port, debug=debug)