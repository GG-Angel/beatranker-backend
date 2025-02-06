import numpy as np
import pandas as pd
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.ml.models import apply_new_modifiers, generate_plot
from app.models.messages import ModRequest, ModResponse
from app.limiter import limiter
from app.utils.utils import df_to_dict

router = APIRouter()

@router.put("/modifiers", response_model=ModResponse)
@limiter.limit("30/minute")
async def modify_recommendations(request: Request, data: ModRequest):
  print("[Modify]: Parsing request...")
  recs_df = pd.DataFrame([row.model_dump() for row in data.recs])  
  model = np.array(data.model)
  new_mods = data.mods

  # update scores according to new modifiers
  print("[Modify]: Predicting scores with new modifiers", new_mods)
  mod_df = apply_new_modifiers(model, recs_df, new_mods)
  print("[Modify] Predictions complete!")

  resp_dict = {
    "recs": df_to_dict(mod_df),
    "plot": generate_plot(mod_df)
  }

  return JSONResponse(resp_dict)