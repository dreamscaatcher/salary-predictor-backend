from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from app.predict import PredictionFeatures, predict_salary

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
async def root():
    return {"message": "Welcome to the Data Science Salary Predictor API"}

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
    uvicorn.run(app, host="0.0.0.0", port=8000)
