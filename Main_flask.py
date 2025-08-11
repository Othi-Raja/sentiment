from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pickle
# Initialize Flask app
app = Flask(__name__)
# Enable CORS for all routes (allow Chrome extension to connect)
CORS(app, resources={r"/*": {"origins": "*"}})
# Load sentiment model
SentimendModel = pickle.load(open("sentimmodel.pkl", "rb"))
# Home page
@app.route("/")
def HomePage():
    return render_template("index.html")
# Old form-based route
@app.route("/result.html", methods=['POST'])
def result():
    data = request.form['inputText']
    prediction = SentimendModel([data])
    result = prediction[0]
    label = result['label']
    score = result['score']
    return render_template('result.html', Label=label, Score=score, data2=data)
# API endpoint for Chrome extension
@app.route("/analyze", methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        text = data.get("text", "").strip()
        if not text:
            return jsonify({"error": "No text provided"}), 400
        prediction = SentimendModel([text])
        result = prediction[0]
        label = result['label']
        score = result['score']
        return jsonify({
            "label": label,
            "score": score,
            "text": text
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
if __name__ == "__main__":
    # Make sure this runs on the same host/port as in your Chrome extension
    app.run(host='127.0.0.1', port=3001, debug=True)
