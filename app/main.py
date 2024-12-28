from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd

app = FastAPI()

# schema
class Recommendation(BaseModel):
  map_name: str
  star_rating: float
  predicted_pp: float

@app.get("/recommendations", response_model=List[Recommendation])
async def get_recommendations(playerId: str):
  # logic here

  # example data
  data = [{"map_name": "Map 1", "star_rating": 4.5, "predicted_pp": 280},
          {"map_name": "Map 2", "star_rating": 5.0, "predicted_pp": 320}]
  return data
