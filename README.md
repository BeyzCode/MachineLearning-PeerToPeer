# Documentation for Machine Learning and Docker Implementation

## Section 1: Machine Learning (IPYNB)

### Overview

This section describes the workflow for training a stacking ensemble model using predictions from three Support Vector Classifiers (SVCs) and a meta-model. The stacking model is implemented in a Jupyter Notebook (.ipynb).

### Prerequisites

- Python 3.x
- Jupyter Notebook
- Required Python libraries:
  - numpy
  - pandas
  - scikit-learn
  - joblib

Install the required libraries:

```bash
pip install numpy pandas scikit-learn joblib
```

### Workflow

1. **Data Preparation**:

   - Load your dataset and split it into training and testing sets.
   - Example:
     ```python
     from sklearn.model_selection import train_test_split
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
     ```

2. **Train Node Models**:

   - Train three SVC models (node models) on the training data.
   - Save each model using `joblib`.

   ```python
   from sklearn.svm import SVC
   from joblib import dump

   svc1 = SVC(probability=True).fit(X_train, y_train)
   dump(svc1, "svc1_model.pkl")
   ```

3. **Generate Meta-Features**:

   - Use predictions from the node models to create meta-features for training the meta-model.

   ```python
   meta_features_train = np.column_stack([
       svc1.predict(X_train),
       svc2.predict(X_train),
       svc3.predict(X_train)
   ])
   ```

4. **Train the Meta-Model**:

   - Train a classifier (e.g., Logistic Regression) using the meta-features.

   ```python
   from sklearn.linear_model import LogisticRegression

   meta_model = LogisticRegression().fit(meta_features_train, y_train)
   dump(meta_model, "stacking_model.pkl")
   ```

5. **Evaluation**:

   - Evaluate the meta-model on the test set.

   ```python
   from sklearn.metrics import classification_report

   print(classification_report(y_test, meta_model.predict(meta_features_test)))
   ```

6. **Export Meta-Model**:
   - Save the trained meta-model as `stacking_model.pkl` for later use.

### Output

- Trained node models: `svc1_model.pkl`, `svc2_model.pkl`, `svc3_model.pkl`
- Trained stacking model: `stacking_model.pkl`

---

## Section 2: Docker Implementation

### Overview

This section provides a guide for deploying a distributed prediction system using Flask and Docker. The architecture includes three nodes (SVC models) and a central Flask server for aggregating predictions.

### Prerequisites

- Docker
- Python 3.x
- Flask
- Required Python libraries:
  - flask
  - requests
  - joblib
  - numpy

Install the required libraries:

```bash
pip install flask requests joblib numpy
```

### Project Structure

```
project/
|-- node1/
|   |-- app.py
|   |-- svc1_model.pkl
|   |-- Dockerfile
|
|-- node2/
|   |-- app.py
|   |-- svc2_model.pkl
|   |-- Dockerfile
|
|-- node3/
|   |-- app.py
|   |-- svc3_model.pkl
|   |-- Dockerfile
|
|-- central_server/
|   |-- app.py
|   |-- stacking_model.pkl
|   |-- Dockerfile
|
|-- docker-compose.yml
```

### Implementation Steps

#### Node Applications

Each node serves predictions from its respective SVC model.

**app.py**:

```python
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load("svc1_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = data['features']
    prediction = model.predict([features])
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

**Dockerfile**:

```dockerfile
FROM python:3.9
WORKDIR /app
COPY . /app
RUN pip install flask joblib
EXPOSE 5000
CMD ["python", "app.py"]
```

#### Central Server

The central server aggregates predictions from all nodes and generates a final prediction using the stacking model.

**app.py**:

```python
from flask import Flask, request, jsonify
import requests
import joblib
import numpy as np

NODE_URLS = {
    "node1": "http://node1:5000/predict",
    "node2": "http://node2:5000/predict",
    "node3": "http://node3:5000/predict"
}

app = Flask(__name__)
stacking_model = joblib.load("stacking_model.pkl")

@app.route('/aggregate_predict', methods=['POST'])
def aggregate_predict():
    data = request.get_json()
    features = data['features']
    predictions = []

    for url in NODE_URLS.values():
        response = requests.post(url, json={'features': features})
        predictions.append(response.json()['prediction'][0])

    meta_features = np.array(predictions).reshape(1, -1)
    final_prediction = stacking_model.predict(meta_features)
    return jsonify({'final_prediction': final_prediction.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

**Dockerfile**:

```dockerfile
FROM python:3.9
WORKDIR /app
COPY . /app
RUN pip install flask requests joblib numpy
EXPOSE 5000
CMD ["python", "app.py"]
```

#### Docker Compose

Use Docker Compose to manage all containers.

**docker-compose.yml**:

```yaml
version: "3.8"
services:
  node1:
    build: ./node1
    ports:
      - "5001:5000"
  node2:
    build: ./node2
    ports:
      - "5002:5000"
  node3:
    build: ./node3
    ports:
      - "5003:5000"
  central_server:
    build: ./central_server
    ports:
      - "5000:5000"
    depends_on:
      - node1
      - node2
      - node3
```

### Steps to Run

1. **Build and Run Containers**:

   ```bash
   docker-compose up --build
   ```

2. **Test the System**:
   Send a POST request to the central server.

   ```bash
   curl -X POST -H "Content-Type: application/json" \
        -d '{"features": [1, 2, 3, 4, 5, 6, 7, 8]}' \
        http://localhost:5000/aggregate_predict
   ```

3. **Expected Output**:
   ```json
   {
     "final_prediction": ["BENIGN"]
   }
   ```

---

This documentation provides a detailed guide to setting up and running both the machine learning pipeline and the Dockerized deployment system.
