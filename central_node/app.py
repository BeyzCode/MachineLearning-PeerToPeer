from flask import Flask, request, jsonify
import requests
import joblib
import numpy as np

# Load stacking model
stacking_model = joblib.load("meta_model.pkl")

# Node URLs
NODE_URLS = {
    "node1": "http://node1:5000/predict",
    "node2": "http://node2:5000/predict",
    "node3": "http://node3:5000/predict"
}

app = Flask(__name__)

@app.route('/aggregate_predict', methods=['POST'])
def aggregate_predict():
    try:
        # Parse input features
        data = request.get_json()
        features = data['features']  # Fitur asli, diteruskan ke setiap node

        # Collect predictions from each node
        predictions = []
        for node_name, url in NODE_URLS.items():
            response = requests.post(url, json={'features': features})
            node_prediction = response.json()
            predictions.append(node_prediction['prediction'][0])  # Scalar prediction

        # Debugging: Log predictions from nodes
        print(f"Predictions from nodes: {predictions}")

        # Convert predictions to array and reshape for stacking model
        meta_features = np.array(predictions).reshape(1, -1)

        # Debugging: Log input to stacking model
        print(f"Input to stacking model: {meta_features}")

        # Final prediction using stacking model
        final_prediction = stacking_model.predict(meta_features)

        return jsonify({'final_prediction': final_prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 400



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
