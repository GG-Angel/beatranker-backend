import json
import time
import requests
import pandas as pd
from datetime import datetime
from app.pp import calc_modified_rating, calc_pp_from_accuracy
from app.utils import clean_song_id, time_ago

# map type conversions
MAP_TYPES = { 
  1: "Accuracy", 
  2: "Tech", 
  4: "Midspeed", 
  8: "Speed"
}

# split map ratings
RATINGS = ["passRating", "accRating", "techRating"]

class APIError(Exception):
  """ Custom exception for API-related errors. """
  pass

def fetch_scores(player_id: str) -> pd.DataFrame:
  """ Gets every ranked score set by a player on BeatLeader.

  Args:
    playerId (str): The player's BeatLeader id

  Returns:
    scores_df (df): Every ranked score sorted by descending PP

  Raises:
    APIError: If a call to the BeatLeader API fails or returns a non-200 status code
  """

  # datapoints of interest
  song_keys = ['name', 'subName', 'author', 'mapper', 'bpm', 'duration']
  difficulty_keys = ["stars", "passRating", "accRating", "techRating", "modifiersRating", "difficultyName", "type"]
  score_keys = ["accuracy", "pp", "rank", "modifiers", "fullCombo"]

  score_rows = []

  page = 1
  while True:
    url = f"https://api.beatleader.xyz/player/{player_id}/scores?sortBy=pp&order=desc&page={page}&count=10&type=ranked"
    resp = requests.get(url)
    if (resp.status_code != 200): 
      raise APIError(f"Failed to fetch scores for {player_id}. Response: {resp.text}")

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
        "songId":        clean_song_id(song["id"]),
        "cover":         song["coverImage"],
        "fullCover":     song["fullCoverImage"]
      }
      song_data  = { key: song[key] for key in song_keys }
      diff_data  = { key: difficulty[key] for key in difficulty_keys }
      score_data = { key: score[key] for key in score_keys }

      # apply modifiers
      modifiers = score_data["currentMods"] = score_data["predictedMods"] = score["modifiers"].split(",") if score["modifiers"] != "" else []
      map_mod_ratings = diff_data["modifiersRating"]
      for rating in RATINGS:
        base_rating = diff_data[rating]
        modified_rating = calc_modified_rating(base_rating, rating, map_mod_ratings, modifiers)
        diff_data[f"{rating}Mod"] = modified_rating
      diff_data[f"starsMod"] = calc_pp_from_accuracy(0.96, *[diff_data[rating] for rating in RATINGS])["total_pp"] / 52

      # convert map type and time set
      diff_data["type"] = MAP_TYPES.get(diff_data["type"], "Unknown")
      score_data["timePost"] = score["timepost"]
      date_set = score_data["dateSet"] = datetime.fromtimestamp(score["timepost"])
      score_data["timeAgo"] = time_ago(date_set)

      # append score row
      score_rows.append({**metadata, **song_data, **diff_data, **score_data})

    page += 1
    time.sleep(0.2)

  # create full dataframe
  scores_df = pd.DataFrame(score_rows)

  return scores_df

def fetch_profile(player_id: str) -> dict:
  """ Gets general information about a player on BeatLeader. 
  
  Params:
    player_id (str): The player's BeatLeader id

  Returns:
    player_data (dict): Contains id, names, avatar, country, pp, and rank

  Raises:
    APIError: If a call to the BeatLeader API fails or returns a non-200 status code
  """
  
  info_keys = ["id", "name", "alias", "avatar", "country", "pp", "rank", "countryRank"]
  
  url = f"https://api.beatleader.xyz/player/{player_id}"
  resp = requests.get(url)
  if resp.status_code != 200:
    raise APIError(f"Failed to fetch profile. Response: {resp.text}")
  
  data = resp.json()

  return { key: data[key] for key in info_keys }

def fetch_maps() -> pd.DataFrame:
  """ Gets every ranked map that exists on BeatLeader. 
  
  Returns:
    maps_df (df): Contains song and difficulty information

  Raises:
    APIError: If a call to the BeatLeader API fails or returns a non-200 status code
  """

  # datapoints of interest
  song_keys = ['name', 'subName', 'author', 'mapper', 'bpm', 'duration']
  difficulty_keys = ["stars", "passRating", "accRating", "techRating", "modifiersRating", "difficultyName", "type"]
  
  map_rows = []

  page = 1
  while True:
    # fetch data from beatleader api
    url = f"https://api.beatleader.xyz/maps?page={page}&count=10&type=ranked"
    resp = requests.get(url)
    if (resp.status_code != 200): 
      raise APIError(f"Failed to fetch ranked maps. Response: {resp.text}") 

    maps = resp.json()["data"]
    if not maps: break

    for ranked_map in maps:
      metadata = {
        "leaderboardId": "",
        "songId":    clean_song_id(ranked_map["id"]),
        "cover":     ranked_map["coverImage"],
        "fullCover": ranked_map["fullCoverImage"]
      }
      song_data = { key: ranked_map[key] for key in song_keys }

      for difficulty in ranked_map["difficulties"]:
        metadata["leaderboardId"] = difficulty["leaderboardId"]
        diff_data = { key: difficulty[key] for key in difficulty_keys }
        
        # mod ratings stay the same since there are no mods
        for rating in ["stars"] + RATINGS:
          diff_data[f"{rating}Mod"] = diff_data[rating]

        # convert map type
        diff_data["type"] = MAP_TYPES.get(diff_data["type"], "Unknown")

        # append map row
        map_rows.append({**metadata, **song_data, **diff_data})

    page += 1
    time.sleep(0.2)

  # create full dataframe (drop non-ranked maps)
  maps_df = pd.DataFrame(map_rows).dropna(subset=["stars"])

  return maps_df