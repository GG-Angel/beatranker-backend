import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
from plotly.utils import PlotlyJSONEncoder
from sklearn.preprocessing import PolynomialFeatures
from fetcher import RATINGS
from pp import WEIGHT_CURVE, calc_modified_rating, calc_pp_from_accuracy
from utils import filter_unplayed

# data given by predictions
PRED_FEATURES = ['leaderboardId', 'songId', 'cover', 'fullCover', 'name', 'subName',
                 'author', 'mapper', 'bpm', 'duration', 'difficultyName', 'type',
                 'stars', 'passRating', 'accRating', 'techRating', 
                 'starsMod', 'passRatingMod', 'accRatingMod', 'techRatingMod', "modifiersRating",
                 "status", "rank", "timeAgo", "timePost", "currentMods", "predictedMods", "isFiltered",
                 "currentAccuracy", "predictedAccuracy", "accuracyGained",
                 "currentPP", "predictedPP", "maxPP", "unweightedPPGain", "weightedPPGain", "weight"]

def train_model(scores_df: pd.DataFrame) -> np.array:
  """ Trains an exponential regression model on the player's existing scores, which predicts accuracy using a map's difficulty ratings. 
  
  Params:
    scores_df (df): The player's existing ranked scores

  Returns:
    model (arr): 1d array where the coef. correspond to the polynomial X-features; includes a bias coef. at index 0 for the y-intercept.
  """

  # calculate days since scores were set
  max_date = scores_df["dateSet"].max()
  days_since = (max_date - scores_df["dateSet"]).dt.days

  # apply weighted decay function so newer scores have more influence
  lambda_value = 0.1
  decay_weights = np.exp(-lambda_value * days_since / 14) # 2 weeks
  
  # modified ratings as independent variables
  X = scores_df[["passRatingMod", "accRatingMod", "techRatingMod"]].values.reshape(-1, 3)
  X_poly = PolynomialFeatures(degree=2, include_bias=False).fit_transform(X)

  # dependent variable; invert to mimic downward curve
  y_inv = (1 - scores_df["accuracy"]).to_numpy().reshape(-1, 1)
  y_inv_log = np.log(y_inv)

  # set up lobf equation
  W = np.diag(decay_weights)
  X_poly_bias = np.column_stack([np.ones(X_poly.shape[0]), X_poly])
  XtW = np.matmul(X_poly_bias.T, W)
  XtWX_inv = np.linalg.inv(np.matmul(XtW, X_poly_bias))
  XtWy = np.matmul(XtW, y_inv_log)

  # generate model
  model = np.matmul(XtWX_inv, XtWy)

  return model

def apply_weight_curve(pred_df: pd.DataFrame) -> pd.DataFrame:
  """ Applies the BeatLeader weight curve to the predicted scores to get the actual pp earned for that score. 
  
  Params:
    pred_df (df): Contains both played and unplayed maps

  Returns:
    weighted_df (df): The same dataframe, but with columns for weighted pp gain and the weights applied
  """

  # sort by descending pp
  weighted_df = pred_df.copy()  
  weighted_df = weighted_df.sort_values(by="maxPP", ascending=False)
  current_pp = weighted_df["currentPP"].sort_values(ascending=False).to_numpy()

  # set up weight scaler
  weights = np.zeros(len(pred_df))
  weight_idx = 0
  curve_idx = 0

  # apply the weights according to where they would fit in the player's current scores
  for _, row in weighted_df.iterrows():
    if curve_idx == len(WEIGHT_CURVE) - 1: break
    if row["maxPP"] < current_pp[curve_idx]: curve_idx += 1
    weights[weight_idx] = WEIGHT_CURVE[curve_idx]
    weight_idx += 1

  # add weighted data
  weighted_df["weightedPPGain"] = weighted_df["unweightedPPGain"] * weights
  weighted_df["weight"] = weights

  return weighted_df

def predict_scores(model: np.array, scores_df: pd.DataFrame, maps_df: pd.DataFrame) -> pd.DataFrame:
  """ Generates the player's potential accuracy and pp on every ranked map using the ML model. 
  
  Params:
    model (arr): 1d array where the coef. correspond to the polynomial X-features; includes a bias coef. at index 0 for the y-intercept
    scores_df (df): The player's existing scores
    maps_df (df): All ranked maps on BeatLeader

  Returns:
    pred_df (df): Contains map information, difficulty, and prediction metrics
  """
  
  # distinguish maps the player has and hasn't yet played
  unplayed_df = filter_unplayed(scores_df, maps_df)
  scores_df["status"] = "played"; unplayed_df["status"] = "unplayed"
  pred_df = pd.concat([scores_df, unplayed_df], ignore_index=True)

  # set up model input features
  X = pred_df[["passRatingMod", "accRatingMod", "techRatingMod"]].values.reshape(-1, 3)
  X_poly = PolynomialFeatures(degree=2, include_bias=False).fit_transform(X)

  # get accuracy prediction metrics
  pred_df["currentAccuracy"] = np.where(~pred_df["accuracy"].isna(), pred_df["accuracy"], 0)
  pred_df["predictedAccuracy"] = np.minimum(1, 1 - np.exp(np.dot(X_poly, model[1:]) + model[0]))
  pred_df["accuracyGained"] = np.maximum(0, pred_df["predictedAccuracy"] - pred_df["currentAccuracy"])

  # get pp prediction metrics
  pred_df["currentPP"] = np.where(~pred_df["pp"].isna(), pred_df["pp"], 0)
  pred_df["predictedPP"] = pred_df.apply(lambda row: calc_pp_from_accuracy(
    row["predictedAccuracy"], row["passRatingMod"], row["accRatingMod"], row["techRatingMod"])["total_pp"], 
    axis=1)
  pred_df["maxPP"] = np.maximum(pred_df["currentPP"], pred_df["predictedPP"])
  pred_df["isFiltered"] = False

  # generate pp gain per map
  pred_df["unweightedPPGain"] = np.maximum(0, pred_df["predictedPP"] - pred_df["currentPP"])
  pred_df = apply_weight_curve(pred_df)

  # remove empty modifiers
  for col in ["currentMods", "predictedMods"]:
    pred_df[col] = pred_df[col].apply(lambda x: np.nan if not x else x)

  return pred_df[PRED_FEATURES].sort_values(by="weightedPPGain", ascending=False)

def apply_new_modifiers(model: np.array, recs_df: pd.DataFrame, new_mods: list[str]) -> pd.DataFrame:
  """Applies new modifiers to each map and updates their star ratings and predictions.
  
  Params:
    model (arr): 1d array where the coef. correspond to the polynomial X-features; includes a bias coef. at index 0 for the y-intercept
    recs_df (df): Table of recommendations already given by the API
    new_mods (arr): Array of mods to apply to every map

  Returns:
    mod_df (df): The same table but with updated mod ratings and predictions
  """
  
  recs_df["predictedMods"] = ([new_mods] * len(recs_df)) if len(new_mods) > 0 else ([None] * len(recs_df))
  
  for index, level in recs_df.iterrows():
      map_mod_ratings = level["modifiersRating"]
      modified_ratings = []
      for rating in RATINGS:
          base_rating = level[rating]
          modified_rating = calc_modified_rating(base_rating, rating, map_mod_ratings, new_mods)            
          recs_df.at[index, f"{rating}Mod"] = modified_rating
          modified_ratings.append(modified_rating) 
      recs_df.at[index, "starsMod"] = calc_pp_from_accuracy(0.96, *modified_ratings)["total_pp"] / 52

  # set up model input features
  X = recs_df[["passRatingMod", "accRatingMod", "techRatingMod"]].values.reshape(-1, 3)
  X_poly = PolynomialFeatures(degree=2, include_bias=False).fit_transform(X)

  # get accuracy prediction metrics
  recs_df["predictedAccuracy"] = np.minimum(1, 1 - np.exp(np.dot(X_poly, model[1:]) + model[0]))
  recs_df["accuracyGained"] = np.maximum(0, recs_df["predictedAccuracy"] - recs_df["currentAccuracy"])

  # get pp prediction metrics
  recs_df["predictedPP"] = recs_df.apply(lambda row: calc_pp_from_accuracy(
    row["predictedAccuracy"], row["passRatingMod"], row["accRatingMod"], row["techRatingMod"])["total_pp"], 
    axis=1)
  recs_df["maxPP"] = np.maximum(recs_df["currentPP"], recs_df["predictedPP"])

  # generate pp gain per map
  recs_df["unweightedPPGain"] = np.maximum(0, recs_df["predictedPP"] - recs_df["currentPP"])
  recs_df = apply_weight_curve(recs_df)
  
  return recs_df

import numpy as np

def generate_plot(recs_df: pd.DataFrame):
    plot = px.scatter(
      recs_df, x="starsMod", y="predictedAccuracy", color="type", 
      hover_data=[
        "currentAccuracy", "predictedMods", "predictedPP", "weightedPPGain", "status", 
        "name", "mapper", "type", "difficultyName", "passRatingMod", "accRatingMod", "techRatingMod"], 
      title="Accuracy Potential against Overall Star Rating"
    )
    plot_json = json.loads(plot.to_json())    
    return plot_json