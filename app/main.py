import os
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Any, List, Dict, Optional
from fastapi.responses import JSONResponse
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.fetcher import fetch_maps, fetch_profile, fetch_scores
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

maps_df = pd.DataFrame()
maps_time = datetime.now(timezone.utc)

# fetches ranked maps from beatleader api and loads them into memory
async def get_ranked_maps():
  global maps_df, maps_time
  try:
    print("[Maps] Fetching ranked maps...")
    maps_df = await fetch_maps()
    maps_time = datetime.now(timezone.utc)
    print("[Maps] Refresh complete!")
  except:
    print("[Maps] Failed to refresh ranked maps.")

# fetch ranked maps on startup and set task to refresh
@app.on_event("startup")
async def startup_event():
  # await get_ranked_maps()

  # REMOVE WHEN DONE
  global maps_df
  script_dir = os.path.dirname(os.path.abspath(__file__))
  csv_path = os.path.join(script_dir, "ranked_maps.csv")
  maps_df = pd.read_csv(csv_path)

  asyncio.create_task(refresh_maps())

# refresh ranked maps every 2 hours
async def refresh_maps():
  while True:
    await asyncio.sleep(2 * 60 * 60) # 2 hours
    await get_ranked_maps()

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
  isFiltered: bool
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
  plot: Dict[str, Any]
  lastMapRefresh: str

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
    player_dict = await fetch_profile(player_id)

    print(f"[{player_id}] Fetching scores...")
    scores_df = await fetch_scores(player_id)
  except:
    raise HTTPException(status_code=500, detail="Failed to fetch player data.")

  print(f"[{player_id}] Predicting scores...")
  model = train_model(scores_df)
  recs_df = predict_scores(model, scores_df, maps_df)
  top_play = scores_df.loc[scores_df["pp"].idxmax()]
  print(f"[{player_id}] Predictions complete!")

  resp_dict = { 
    "profile": {
      **player_dict,
      "bestPP": float(top_play["pp"]),
      "bestRank": int(top_play["rank"]),
      "medianPP": float(scores_df["pp"].median()),
      "medianRank": int(scores_df["rank"].median())
    }, 
    "ml": {
      "model": model.tolist(),
      "plot": generate_plot(recs_df),
      "lastMapRefresh": maps_time.isoformat()
    },
    "recs": df_to_dict(recs_df),
  }
  
  return JSONResponse(resp_dict)

@app.put("/modifiers", response_model=List[Recommendation])
async def modify_recommendations(data: RecommendationsMod):
  print("[Modify]: Parsing request...")
  recs_df = pd.DataFrame([row.model_dump() for row in data.recs])  
  model = np.array(data.model)
  new_mods = data.mods

  # update scores according to new modifiers
  print("[Modify]: Predicting scores with new modifiers", new_mods)
  mod_df = apply_new_modifiers(model, recs_df, new_mods)
  mod_dict = df_to_dict(mod_df)

  return JSONResponse(mod_dict)