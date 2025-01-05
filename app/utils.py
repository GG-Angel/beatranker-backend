import json
import requests
import pandas as pd
from datetime import datetime

def is_valid_id(player_id: str) -> bool:
  url = f"https://api.beatleader.xyz/player/{player_id}/exists"
  resp = requests.get(url)
  return resp.status_code == 200

def clean_song_id(song_id: str) -> str:
  if 'x' in song_id:
    return song_id[:song_id.index('x')]
  return song_id

def filter_unplayed(scores_df: pd.DataFrame, maps_df: pd.DataFrame) -> pd.DataFrame: 
  bool_unplayed = ~maps_df["leaderboardId"].isin(scores_df["leaderboardId"])
  return maps_df[bool_unplayed].copy()

def df_to_json(df: pd.DataFrame) -> list[dict]:
  return json.loads(df.to_json(orient="records"))

def dict_to_json(dict: dict) -> str:
  return json.loads(json.dumps(dict))

def time_ago(dt: datetime) -> str:
  now = pd.to_datetime(datetime.now())
  diff = now - pd.to_datetime(dt)

  days = diff.days
  seconds = diff.seconds
  months = (now.year - dt.year) * 12 + now.month - dt.month
  years = now.year - dt.year

  if years > 0:
    return f"{years} year{'s' if years > 1 else ''} ago"
  elif months > 0:
    return f"{months} month{'s' if months > 1 else ''} ago"
  elif days > 7:
    weeks = days // 7
    return f"{weeks} week{'s' if weeks > 1 else ''} ago"
  elif days > 1:
    return f"{days} day{'s' if days > 1 else ''} ago"
  elif seconds >= 3600:
    hours = seconds // 3600
    return f"{hours} hour{'s' if hours > 1 else ''} ago"
  elif seconds >= 60:
    minutes = seconds // 60
    return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
  else:
    return "Just now"