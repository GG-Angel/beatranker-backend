from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from cachetools import TTLCache

from ml.models import generate_plot, predict_scores, train_model
from models.messages import AllResponse
from services.fetcher import fetch_profile, fetch_scores
from utils.utils import df_to_dict, is_valid_id
from limiter import limiter
from services.maps import get_cached_maps, get_last_map_refresh

router = APIRouter(tags=["Recommendations"])

cache = TTLCache(maxsize=100, ttl=600) # 10 minutes

@router.get("/recommendations/{player_id}", response_model=AllResponse)
@limiter.limit("5/minute")
async def get_recommendations(
  request: Request,
  player_id: str,
  force: bool = Query(False, description="Force a fresh fetch, bypassing the cache."),
):
  """
  Get recommendations for a given player based on existing scores and performance metrics.
  """

  if not is_valid_id(player_id):
    raise HTTPException(status_code=404, detail="Player does not exist.")
  
  if player_id in cache and not force:
    return cache[player_id]
  
  try:
    print(f"[{player_id}] Fetching profile...")
    player_dict = await fetch_profile(player_id)

    print(f"[{player_id}] Fetching scores...")
    scores_df = await fetch_scores(player_id)
  except Exception as e:
    raise HTTPException(status_code=500, detail=f"Failed to fetch player data. {str(e)}")

  print(f"[{player_id}] Predicting scores...")
  model = train_model(scores_df)
  maps_df = get_cached_maps()
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
      "lastMapRefresh": get_last_map_refresh().isoformat()
    },
    "recs": df_to_dict(recs_df),
  }

  cache[player_id] = resp_dict
  
  return JSONResponse(resp_dict)