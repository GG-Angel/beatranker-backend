from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from app.models.models import ProfileCompact
from app.services.fetcher import fetch_players
from app.services.limiter import limiter

router = APIRouter(tags=["Search"])

@router.get("/search/", response_model=List[ProfileCompact])
@limiter.limit("30/minute")
async def search_players(
  request: Request, 
  query: Optional[str] = Query("", max_length=100),
  k: Optional[int] = Query(5, ge=1, le=50)  # Default 5, min 1, max 50
):
  """
  Search for players by username and return the top k results sorted by rank.
  """

  try:
    players = await fetch_players(query, k)
    return JSONResponse(players)
  except Exception as e:
    raise HTTPException(status_code=500, detail=f"Failed to search players: {str(e)}")