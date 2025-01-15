from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from app.predict import PredictionFeatures, predict_salary
import os

app = FastAPI()

# Add CORS middleware with proper configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://salary-predictor-frontend-production.up.railway.app",
        "http://localhost:3000"  # Keep for local development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Welcome to the Data Science Salary Predictor API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/predict")
async def predict(features: PredictionFeatures):
    try:
        print(f"Received prediction request with features: {features}")
        prediction = predict_salary(features)
        print(f"Successfully generated prediction: {prediction}")
        return {"predicted_salary_usd": prediction}
    except Exception as e:
        print(f"Error processing prediction request: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate prediction: {str(e)}"
        )

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)