# from fastapi import FastAPI
# from pydantic import BaseModel,Field
# import numpy as np
# import joblib
# from typing import Annotated

# #  Load the model
# # with open(r"house_model.pkl", "rb") as file:
# #     model = pickle.load(file)
# model = joblib.load("house.pkl")

# app = FastAPI()

# # Data validation of model
# class HouseData(BaseModel):
#     area: float = Field(..., description="Enter area in square feet", example=3000)
#     bedrooms: int = Field(..., description="Enter number of bedrooms", example=3)
#     bathrooms: int = Field(..., description="Enter number of bathrooms", example=2)
#     stories: int = Field(..., description="Enter number of stories", example=1)
#     parking: int = Field(..., description="Number of parking spots needed", example=1)
#     mainroad_yes: int = Field(..., description="1 if main road access is required, else 0", example=1)
#     guestroom_yes: int = Field(..., description="1 if guestroom is needed, else 0", example=0)
#     basement_yes: int = Field(..., description="1 if basement is needed, else 0", example=1)
#     hotwaterheating_yes: int = Field(..., description="1 if hot water heating is required, else 0", example=0)
#     airconditioning_yes: int = Field(..., description="1 if air conditioning is required, else 0", example=1)
#     prefarea_yes: int = Field(..., description="1 if located in preferred area, else 0", example=0)
#     furnishingstatus_semi_furnished: int = Field(..., description="1 if semi-furnished, else 0", example=1)
#     furnishingstatus_unfurnished: int = Field(..., description="1 if unfurnished, else 0", example=0)
    
    


    
    
#     #   endpoint
# @app.post("/predict")
# def predict_price(data: HouseData):
#     # Convert input to numpy array
#     input_data = np.array([[ 
#         data.area,
#         data.bedrooms,
#         data.bathrooms,
#         data.stories,
#         data.parking,
#         data.mainroad_yes,
#         data.guestroom_yes,
#         data.basement_yes,
#         data.hotwaterheating_yes,
#         data.airconditioning_yes,
#         data.prefarea_yes,
#         data.furnishingstatus_semi_furnished,
#         data.furnishingstatus_unfurnished
#     ]])
    
#     # prediction
#     prediction = model.predict(input_data)[0]
#     return {"predicted_price": f"{round(prediction, 2)} rupees"}



from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("house1.pkl")

app = FastAPI()

# Define the expected input data structure (raw values, not one-hot)
class HouseData(BaseModel):
    area: float = Field(..., description="Enter area in square feet", example=3000)
    bedrooms: int = Field(..., description="Enter number of bedrooms", example=3)
    bathrooms: int = Field(..., description="Enter number of bathrooms", example=2)
    stories: int = Field(..., description="Enter number of stories", example=1)
    parking: int = Field(..., description="Number of parking spots needed", example=1)
    mainroad: str = Field(..., description="yes or no", example="yes")
    guestroom: str = Field(..., description="yes or no", example="no")
    basement: str = Field(..., description="yes or no", example="yes")
    hotwaterheating: str = Field(..., description="yes or no", example="no")
    airconditioning: str = Field(..., description="yes or no", example="yes")
    prefarea: str = Field(..., description="yes or no", example="no")
    furnishingstatus: str = Field(..., description="furnished, semi-furnished, or unfurnished", example="semi-furnished")

# Root endpoint (optional)
@app.get("/")
def read_root():
    return {"message": "🏡 House Price Prediction API is running"}

# Prediction endpoint
@app.post("/predict")
def predict_price(data: HouseData):
    # Convert input to DataFrame
    input_df = pd.DataFrame([data.dict()])

    # Make prediction
    prediction = model.predict(input_df)[0]

    # Return result
    return {"predicted_price": f"{round(prediction, 2)} rupees"}