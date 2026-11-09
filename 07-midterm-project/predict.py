import pickle
import xgboost as xgb
from fastapi import FastAPI
from pydantic import BaseModel

# 1. Load the model and vectorizer from the single .bin file
with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = FastAPI()

# 2. Define the expected data format using Pydantic
class HouseFeatures(BaseModel):
    X1: float
    X2: float
    X3: float
    X4: float
    X5: float
    X6: int
    X7: float
    X8: int

@app.get("/")
def home():
    return {"message": "Heating Load Prediction API is running!"}

# 3. Create the prediction endpoint
@app.post("/predict")
def predict(features: HouseFeatures):
    # Convert the incoming Pydantic object to a standard Python dictionary
    house_dict = features.model_dump()
    
    # Transform the dictionary into the format the model expects (a list of dicts)
    X = dv.transform([house_dict])
    
    # XGBoost requires feature names for the DMatrix
    feature_names = dv.get_feature_names_out().tolist()
    dmatrix = xgb.DMatrix(X, feature_names=feature_names)
    
    # Make the prediction
    y_pred = model.predict(dmatrix)
    
    # Return the result as JSON
    return {"heating_load_prediction": float(y_pred[0])}