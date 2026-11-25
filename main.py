from fastapi import FastAPI
from pydantic import BaseModel
import joblib
app = FastAPI()
model = joblib.load("model.joblib")
class Features(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
@app.get("/")
def root():
    return {"message": "ML API is running"}
@app.post("/predict")
def predict(f: Features):
    pred = model.predict([[f.sepal_length, f.sepal_width, f.petal_length, f.petal_width]])
    return {"prediction": int(pred[0])}