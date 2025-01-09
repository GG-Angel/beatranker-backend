import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from app.pp import WEIGHT_CURVE, calc_pp_from_accuracy
from app.utils import filter_unplayed

# data given by predictions
PRED_FEATURES = ['leaderboardId', 'songId', 'cover', 'fullCover', 'name', 'subName',
                 'author', 'mapper', 'bpm', 'duration', 'durationMod', 'difficultyName', 'type',
                 'stars', 'passRating', 'accRating', 'techRating', 
                 'starsMod', 'passRatingMod', 'accRatingMod', 'techRatingMod', 
                 "status", "rank", "timeAgo", "timePost", "currentMods", "predictedMods",
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

  # generate pp gain per map
  pred_df["unweightedPPGain"] = np.maximum(0, pred_df["predictedPP"] - pred_df["currentPP"])
  pred_df = apply_weight_curve(pred_df)

  # remove empty modifiers
  for col in ["currentMods", "predictedMods"]:
    pred_df[col] = pred_df[col].apply(lambda x: np.nan if not x else x)

  return pred_df[PRED_FEATURES].sort_values(by="weightedPPGain", ascending=False)