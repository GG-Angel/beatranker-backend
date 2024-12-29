import numpy as np
import pandas as pd
from pp import calc_pp_from_accuracy, weight_curve

RELEVANT_KEYS = [
  'leaderboardId', 'downloadId', 'cover', 'fullCover', 
  'name', 'subName', 'author', 'mapper', 'bpm', 'duration', 'type', 'dateset',
  
]

def filter_unplayed(scores_df, maps_df): 
  bool_unplayed = ~maps_df["leaderboardId"].isin(scores_df["leaderboardId"])
  return maps_df[bool_unplayed]

def predict_score(level, vector, model_dict):
  """ Predict performance on a played/unplayed map using the given ML model

  Params:
    level (series): The map series itself from fetchers
    vector (1d array): Includes a bias 1 at the front with the rest representing the x-features of the model
    model_dict (dict): Includes the ML model array and whether the map is played/unplayed
  Returns:
    predictions (dict): Contains accuracy and pp predictions for the given level
  """

  # predict accuracy on map
  model = model_dict["model"]
  pred_acc = np.dot(vector, model)

  # reverse inversion for unplayed model
  status = model_dict["type"]
  if status == "unplayed":
    pred_acc = 1 - np.exp(pred_acc)
  
  ratings = level[["passRating", "accRating", "techRating"]]
  pred_dict = calc_pp_from_accuracy(pred_acc, *ratings)

  pred_metrics = {
    "modifiers":          level["modifiers"] if status == "played" else [],
    "status":             status,
    "current_acc":        level["accuracy"] * 100 if status == "played" else 0,
    "predicted_acc":      pred_acc * 100,
    "acc_gain":           (pred_acc - (level["accuracy"] if status == "played" else 0)) * 100,
    "current_pp":         level["pp"] if status == "played" else 0,
    "predicted_pp":       pred_dict["total_pp"],
    "unweighted_pp_gain": max(pred_dict["total_pp"] - (level["pp"] if status == "played" else 0), 0)
  }

  return {
    "id":                 level["id"],
    "name":               level["name"],
    "mapper":             level["mapper"],
    "difficulty":         level["difficultyName"],
    "stars":              level["stars"],
  }



def predict_scores(scores_df: pd.DataFrame, maps_df: pd.DataFrame, improve_dict: dict, unplayed_dict: dict) -> pd.DataFrame:
  """ Predict performance on all ranked maps (both played and unplayed) using the 
  ML models; sort predictions by most potential weighted pp.

  Returns:
    optimal_df (df): Table of predicted performance and improvement metrics 
  """

  improvement_rows = []
  unplayed_rows = []

  # score improvements
  for idx, score in scores_df.iterrows():
    vector = np.insert(improve_dict["X"][idx], 0, 1)
    prediction = predict_score(score, vector, improve_dict)
    improvement_rows.append(prediction)

  # new scores
  for idx, unplayed in filter_unplayed(scores_df, maps_df).iterrows():
    vector = np.array([1, unplayed["stars"]])
    prediction = predict_score(unplayed, vector, unplayed_dict)
    unplayed_rows.append(prediction)

  # create full dataframe
  predictions_df = pd.concat([pd.DataFrame(improvement_rows), pd.DataFrame(unplayed_rows)], ignore_index=True)

  # apply beatleder weight curve
  predictions_df.sort_values(by="predicted_pp", ascending=False)
  weights = np.zeros(len(predictions_df))
  weight_idx = 0

  for idx, row in predictions_df.iterrows():
    if weight_idx < len(weight_curve):
      weights[idx] = weight_curve[weight_idx]
      if row["status"] == "improvement":
        weight_idx += 1

  predictions_df["weighted_pp_gain"] = predictions_df["unweighted_pp_gain"] * weights
  predictions_df["weight"] = weights

  # sort by highest weighted pp gain
  predictions_df = predictions_df.sort_values(by="weighted_pp_gain", ascending=False)

  return predictions_df