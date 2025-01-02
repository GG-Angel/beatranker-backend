import requests
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from app.fetcher import fetch_scores
from app.models import predict_scores, train_model
from app.utils import df_to_json, is_valid_id

app = FastAPI()

# schema
class Recommendation(BaseModel):
  leaderboardId: str
  songId: str
  cover: str
  fullCover: str
  name: str
  subName: Optional[str]
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
  rank: Optional[int]
  timeAgo: Optional[str]
  currentMods: Optional[List[str]]
  predictedMods: Optional[List[str]]
  currentAccuracy: float
  predictedAccuracy: float
  accuracyGained: float
  currentPP: float
  predictedPP: float
  maxPP: float
  unweightedPPGain: float
  weightedPPGain: float
  weight: float

@app.get("/recommendations/{player_id}", response_model=List[Recommendation])
async def get_recommendations(player_id: str):
  if not is_valid_id(player_id):
    raise HTTPException(status_code=404, detail="Player does not exist.")

  try:
    maps_df = pd.read_csv("./app/ranked_maps.csv")
    scores_df = fetch_scores(player_id)
  except:
    raise HTTPException(status_code=500, detail="Failed to fetch scores.")

  model = train_model(scores_df)
  pred_df = predict_scores(model, scores_df, maps_df)

  return df_to_json(pred_df)
