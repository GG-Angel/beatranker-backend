from typing import Any, Dict, List
from pydantic import BaseModel
from app.models import Profile, Recommendation, MLData

class AllResponse(BaseModel):
  profile: Profile
  recs: List[Recommendation]
  ml: MLData

class ModRequest(BaseModel):
  recs: List[Recommendation]
  model: List[List[float]]
  mods: List[str]

class ModResponse(BaseModel):
  recs: List[Recommendation]
  plot: Dict[str, Any]
