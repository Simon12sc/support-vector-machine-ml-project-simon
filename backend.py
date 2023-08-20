from flask import Flask, request, jsonify, render_template  # Import necessary modules
import pickle

app = Flask(__name__)

# Load the machine learning model using pickle
with open("model.pkl", "rb") as model_file:
    svc = pickle.load(model_file)

@app.route('/')
def index():
    return render_template('index.html')    
    
    
@app.route("/predict", methods=['POST'])
def predict():
    try:
        data = request.get_json()  # Get JSON data from the request

        values_list = list(data.values())
        result = svc.predict([values_list])[0]

        return jsonify({
            "success": True,
            "result": int(result),
            "info": "0 for malignant ( breast cancer) and 1 for benign (no breast cancer)"
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route("/check")
def check():
    return "hello"

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)