from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from xai.explanation import generate_explanations

app = FastAPI()

class PredictionRequest(BaseModel):
    text: str
    num_links: int
    has_suspicious_url: bool
    urgency_count: int
    readability_score: float

# Load trained model and vectorizer
model = joblib.load("../models/model_0_logistic_regression.pkl")
tfidf = joblib.load("../models/tfidf_vectorizer.pkl")

@app.post("/predict")
async def predict(request: PredictionRequest):
    # Preprocess input
    features = preprocess_input(request, tfidf)
    
    # Make prediction
    prediction = model.predict([features])[0]
    proba = model.predict_proba([features])[0][1]
    
    # Generate explanations
    explanation = generate_explanations(model, features, request.text)
    
    return {
        "prediction": int(prediction),
        "probability": float(proba),
        "explanations": explanation
    }

def preprocess_input(request, tfidf):
    # Your feature engineering logic here
    pass
