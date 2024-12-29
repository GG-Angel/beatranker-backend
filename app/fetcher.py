import requests
import pandas as pd
from datetime import datetime
from pp import calc_modified_rating, calc_pp_from_accuracy

# map type conversions
MAP_TYPES = { 
  1: "Accuracy", 
  2: "Tech", 
  4: "Midspeed", 
  8: "Speed"
}

# split map ratings
RATINGS = ["passRating", "accRating", "techRating"]

def fetch_scores(player_id: str) -> pd.DataFrame:
  """ Gets every ranked score set by a player on BeatLeader.

  Args:
    playerId (str): The player's BeatLeader id

  Returns:
    scores_df (df): Every ranked score sorted by descending PP
  """

  # datapoints of interest
  song_keys = ['name', 'subName', 'author', 'mapper', 'bpm', 'duration']
  difficulty_keys = ["stars", "passRating", "accRating", "techRating", "difficultyName", "type"]
  score_keys = ["accuracy", "pp", "rank", "modifiers", "fullCombo", "maxCombo"]

  score_rows = []

  page = 1
  while True:
    url = f"https://api.beatleader.xyz/player/{player_id}/scores?sortBy=pp&order=desc&page={page}&count=10&type=ranked"
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
        "songId":        song["id"],
        "cover":         song["coverImage"],
        "fullCover":     song["fullCoverImage"]
      }
      song_data  = { key: song[key] for key in song_keys }
      diff_data  = { key: difficulty[key] for key in difficulty_keys }
      score_data = { key: score[key] for key in score_keys }

      # apply modifiers
      modifiers = score.get("modifiers", "").split(",")
      for rating in RATINGS:
        base_rating = diff_data[rating]
        modified_rating = calc_modified_rating(base_rating, rating, difficulty["modifiersRating"], modifiers)
        diff_data[f"mod_{rating}"] = modified_rating
      diff_data[f"mod_stars"] = calc_pp_from_accuracy(0.96, *[diff_data[rating] for rating in RATINGS])["total_pp"] / 52

      # convert map type and time set
      diff_data["type"] = MAP_TYPES.get(diff_data["type"], "Unknown")
      score_data["dateset"] = datetime.fromtimestamp(score["timepost"])

      # append score row
      score_rows.append({**metadata, **song_data, **diff_data, **score_data})

    page += 1

  # create full dataframe
  scores_df = pd.DataFrame(score_rows)

  return scores_df

def fetch_maps() -> pd.DataFrame:
  """ Gets every ranked map that exists on BeatLeader. """

  # datapoints of interest
  song_keys = ['name', 'subName', 'author', 'mapper', 'bpm', 'duration']
  difficulty_keys = ["stars", "passRating", "accRating", "techRating", "difficultyName", "type"]
  
  map_rows = []

  page = 1
  while True:
    # fetch data from beatleader api
    url = f"https://api.beatleader.xyz/maps?page={page}&count=10&type=ranked"
    resp = requests.get(url)
    if (resp.status_code != 200): break 

    maps = resp.json()["data"]
    if not maps: break

    for ranked_map in maps:
      metadata = {
        "leaderboardId": "",
        "songId":    ranked_map["id"],
        "cover":     ranked_map["coverImage"],
        "fullCover": ranked_map["fullCoverImage"]
      }
      song_data = { key: ranked_map[key] for key in song_keys }

      for difficulty in ranked_map["difficulties"]:
        metadata["leaderboardId"] = difficulty["leaderboardId"]
        diff_data = { key: difficulty[key] for key in difficulty_keys }
        
        # mod ratings stay the same since there are no mods
        for rating in ["stars"] + RATINGS:
          diff_data[f"mod_{rating}"] = diff_data[rating]

        # convert map type
        diff_data["type"] = MAP_TYPES.get(diff_data["type"], "Unknown")

        # append map row
        map_rows.append({**metadata, **song_data, **diff_data})

    page += 1

  # create full dataframe (drop non-ranked maps)
  maps_df = pd.DataFrame(map_rows).dropna(subset=["stars"])

  return maps_df