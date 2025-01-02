FROM python:3.9-slim

WORKDIR /app

COPY app.py .
COPY svc1_node1_model.pkl .

RUN pip install flask numpy joblib scikit-learn

EXPOSE 5000

CMD ["python", "app.py"]
