import json
import pandas as pd

def filter_unplayed(scores_df: pd.DataFrame, maps_df: pd.DataFrame) -> pd.DataFrame: 
  bool_unplayed = ~maps_df["leaderboardId"].isin(scores_df["leaderboardId"])
  return maps_df[bool_unplayed].copy()

def df_to_json(df: pd.DataFrame) -> list[dict]:
  return json.loads(df.to_json(orient="records"))