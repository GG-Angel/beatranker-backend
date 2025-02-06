from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from app.main import limiter
from app.services import fetch_profile, fetch_scores, maps_df, maps_time
from app.utils import df_to_dict, is_valid_id
from app.models import AllResponse
from app.ml import train_model, predict_scores, generate_plot
from cachetools import TTLCache

router = APIRouter()
cache = TTLCache(maxsize=100, ttl=1800)

@router.get("/recommendations/{player_id}", response_model=AllResponse)
@limiter.limit("5/minute")
async def get_recommendations(
  request: Request,
  player_id: str,
  force: bool = Query(False, description="Force a fresh fetch, bypassing the cache."),
):
  if not is_valid_id(player_id):
    raise HTTPException(status_code=404, detail="Player does not exist.")
  
  if player_id in cache and not force:
    return cache[player_id]

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

  cache[player_id] = resp_dict
  
  return JSONResponse(resp_dict)