import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()

# schema
class Recommendation(BaseModel):
  leaderboardId: str
  songId: str
  cover: str
  fullCover: str
  name: str
  subName: str
  author: str
  mapper: str
  bpm: float
  duration: int
  difficultyName: str
  type: str
  stars: float
  passRating: float
  accRating: float
  techRating: float
  starsMod: float
  passRatingMod: float
  accRatingMod: float
  techRatingMod: float
  status: str
  modifiers: list[str]
  currentAccuracy: float
  predictedAccuracy: float
  accuracyGained: float
  currentPP: float
  predictedPP: float
  maxPP: float
  unweightedPPGain: float
  weightedPPGain: float
  weight: float

@app.get("/recommendations", response_model=List[Recommendation])
async def get_recommendations(player_id: str):
  # logic here

  # example data
  data = [{"map_name": "Map 1", "star_rating": 4.5, "predicted_pp": 280},
          {"map_name": "Map 2", "star_rating": 5.0, "predicted_pp": 320}]
  pass
