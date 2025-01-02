FROM python:3.9-slim

WORKDIR /app

COPY app.py .
COPY meta_model.pkl .

RUN pip install flask numpy joblib scikit-learn requests

EXPOSE 5000

CMD ["python", "app.py"]
