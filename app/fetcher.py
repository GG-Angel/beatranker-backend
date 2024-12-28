import requests
import pandas as pd
from datetime import datetime
from pp import calc_modified_rating, calc_pp_from_accuracy

def fetch_scores(beatleaderId: str) -> pd.DataFrame:
  """ Gets every ranked score set by a player on BeatLeader.

  Args:
    playerId (str): The player's BeatLeader id

  Returns:
    scores_df (df): Every ranked score sorted by descending PP
  """

  # datapoints of interest
  song_keys = ['name', 'subName', 'author', 'mapper']
  difficulty_keys = ["stars", "passRating", "accRating", "techRating", "difficultyName", 
                    "type", "njs", "nps", "notes", "bombs", "walls", "maxScore", "duration"]
  score_keys = ["accLeft", "accRight", "baseScore", "modifiedScore", "accuracy", 
                "pp", "passPP", "accPP", "techPP", "rank", 
                "fcAccuracy", "fcPp", "weight", "modifiers", "badCuts", 
                "missedNotes", "bombCuts", "wallsHit", "pauses", "fullCombo", "maxCombo"]
  rating_keys = ["passRating", "accRating", "techRating"]
  
  # map type conversions
  map_types = { 
    1: "Accuracy", 
    2: "Tech", 
    4: "Midspeed", 
    8: "Speed"
  }

  score_rows = []

  page = 1
  while True:
    url = f"https://api.beatleader.xyz/player/{beatleaderId}/scores?sortBy=pp&order=desc&page={page}&count=10&type=ranked"
    resp = requests.get(url)
    if (resp.status_code != 200): break

    scores = resp.json()["data"]
    if not scores: break

    for score in scores:
      # crawl through dictionary
      leaderboard = score["leaderboard"]
      song        = leaderboard["song"]
      difficulty  = leaderboard["difficulty"]
      
      # prepare row data
      metadata = {
        "leaderboardId": leaderboard["id"],
        "downloadId":    song["id"],
        "cover":         song["coverImage"],
        "fullCover":     song["fullCoverImage"]
      }
      song_data  = { key: song[key] for key in song_keys }
      diff_data  = { key: difficulty[key] for key in difficulty_keys }
      score_data = { key: score[key] for key in score_keys }

      # apply modifiers
      modifiers = score.get("modifiers", "").split(",")
      for rating in rating_keys:
        base_rating = diff_data[rating]
        modified_rating = calc_modified_rating(base_rating, rating, difficulty["modifiersRating"], modifiers)
        diff_data[rating] = modified_rating
      diff_data["stars"] = calc_pp_from_accuracy(0.96, *[diff_data[rating] for rating in rating_keys])["total_pp"] / 52

      # convert map type and time set
      diff_data["type"] = map_types.get(diff_data["type"], "Unknown")
      score_data["dateset"] = datetime.fromtimestamp(score["timepost"])

      # append score row
      score_rows.append({**metadata, **song_data, **diff_data, **score_data})

    page += 1

  # create full dataframe
  scores_df = pd.DataFrame(score_rows)

  # ensure numerical columns are floats
  float_cols = ["stars", "passRating", "accRating", "techRating", "accuracy", 
                "pp", "passPP", "accPP", "techPP", "weight", "fcAccuracy", "fcPp", "nps"]
  scores_df[float_cols] = scores_df[float_cols].astype(float)

  return scores_df