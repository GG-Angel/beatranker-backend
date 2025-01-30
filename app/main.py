from fastapi.responses import JSONResponse
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
from typing import List, Dict, Optional

from app.fetcher import fetch_profile, fetch_scores
from app.models import apply_new_modifiers, generate_plot, predict_scores, train_model
from app.utils import df_to_dict, is_valid_id

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "PUT"],
    allow_headers=["*"],
)

# schemas
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
  modifiersRating: Dict[str, float]
  status: str
  rank: Optional[int]
  timeAgo: Optional[str]
  timePost: Optional[int]
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

class Profile(BaseModel):
  id: str
  name: str
  alias: str
  avatar: str
  country: str
  pp: float
  rank: int
  countryRank: int
  bestPP: float
  bestRank: int
  medianPP: float
  medianRank: int

class MLData(BaseModel):
  model: List[List[float]]
  plot: str

class ProfileAndRecommendations(BaseModel):
  profile: Profile
  recs: List[Recommendation]
  ml: MLData

class RecommendationsMod(BaseModel):
  recs: List[Recommendation]
  model: List[List[float]]
  mods: List[str]

@app.get("/recommendations/{player_id}", response_model=ProfileAndRecommendations)
async def get_recommendations(player_id: str):
  if not is_valid_id(player_id):
    raise HTTPException(status_code=404, detail="Player does not exist.")
  
  try:
    print(f"[{player_id}] Fetching profile...")
    player_dict = fetch_profile(player_id)

    print(f"[{player_id}] Fetching scores...")
    maps_df = pd.read_csv("./app/ranked_maps.csv")
    scores_df = fetch_scores(player_id)
  except:
    raise HTTPException(status_code=500, detail="Failed to fetch player data.")

  print(f"[{player_id}] Predicting scores...")
  model = train_model(scores_df)
  recs_df = predict_scores(model, scores_df, maps_df)
  top_play = scores_df.loc[scores_df["pp"].idxmax()]

  resp_dict = { 
    "profile": {
      **player_dict,
      "bestPP": float(top_play["pp"]),
      "bestRank": int(top_play["rank"]),
      "medianPP": float(scores_df["pp"].median()),
      "medianRank": int(scores_df["rank"].median())
    }, 
    "recs": df_to_dict(recs_df),
    "ml": {
      "model": model.tolist(),
      "plot": generate_plot(recs_df)
    }
  }
  
  return JSONResponse(resp_dict)

@app.put("/modifiers", response_model=List[Recommendation])
async def update_modifiers(data: RecommendationsMod):
  print("[Modifier Change]: Parsing request...")
  recs_df = pd.DataFrame([row.model_dump() for row in data.recs])  
  model = np.array(data.model)
  new_mods = data.mods

  # update scores according to new modifiers
  print("[Modifier Change]: Predicting scores with new modifiers", new_mods)
  mod_df = apply_new_modifiers(model, recs_df, new_mods)
  mod_dict = df_to_dict(mod_df)

  return JSONResponse(mod_dict)