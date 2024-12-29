import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from pp import WEIGHT_CURVE, calc_pp_from_accuracy
from utils import filter_unplayed

PRED_FEATURES = ['leaderboardId', 'songId', 'cover', 'fullCover', 'name', 'subName',
                 'author', 'mapper', 'bpm', 'duration', 'difficultyName', 'type',
                 'stars', 'passRating', 'accRating', 'techRating', 
                 'mod_stars','mod_passRating', 'mod_accRating', 'mod_techRating', 
                 "status", "modifiers", "current_acc", "pred_acc", "acc_gain",
                 "current_pp", "pred_pp", "max_pp",
                 "unweighted_pp_gain", "weighted_pp_gain", "weights"]

def train_model(scores_df: pd.DataFrame) -> np.array:
  """ TODO """

  # calculate days since scores were set
  max_date = scores_df["dateset"].max()
  days_since = (max_date - scores_df["dateset"]).dt.days

  # apply weighted decay function so newer scores have more influence
  lambda_value = 0.1
  decay_weights = np.exp(-lambda_value * days_since / 14) # 2 weeks
  
  # modified ratings as independent variables
  X = scores_df[["mod_passRating", "mod_accRating", "mod_techRating"]].values.reshape(-1, 3)
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
  """ TODO """

  weighted_df = pred_df.copy()
  
  weighted_df = weighted_df.sort_values(by="max_pp", ascending=False)
  current_pp = weighted_df["current_pp"].sort_values(ascending=False).to_numpy()

  weights = np.zeros(len(pred_df))
  weight_idx = 0
  curve_idx = 0

  for _, row in weighted_df.iterrows():
    if curve_idx < len(WEIGHT_CURVE) - 1 and row["max_pp"] < current_pp[curve_idx]:
      curve_idx += 1
    weights[weight_idx] = WEIGHT_CURVE[curve_idx]
    weight_idx += 1
  
  weighted_df["weighted_pp_gain"] = weighted_df["unweighted_pp_gain"] * weights
  weighted_df["weights"] = weights

  return weighted_df

def predict_scores(model: np.array, scores_df: pd.DataFrame, maps_df: pd.DataFrame) -> pd.DataFrame:
  """ TODO """
  
  unplayed_df = filter_unplayed(scores_df, maps_df)
  scores_df["status"] = "played"; unplayed_df["status"] = "unplayed"
  pred_df = pd.concat([scores_df, unplayed_df], ignore_index=True)

  X = pred_df[["mod_passRating", "mod_accRating", "mod_techRating"]].values.reshape(-1, 3)
  X_poly = PolynomialFeatures(degree=2, include_bias=False).fit_transform(X)

  pred_df["current_acc"] = np.where(~pred_df["accuracy"].isna(), pred_df["accuracy"], 0)
  pred_df["pred_acc"] = np.minimum(1, 1 - np.exp(np.dot(X_poly, model[1:]) + model[0]))
  pred_df["acc_gain"] = np.maximum(0, pred_df["pred_acc"] - pred_df["current_acc"])

  pred_df["current_pp"] = np.where(~pred_df["pp"].isna(), pred_df["pp"], 0)
  pred_df["pred_pp"] = pred_df.apply(lambda row: calc_pp_from_accuracy(
    row["pred_acc"], row["mod_passRating"], row["mod_accRating"], row["mod_techRating"])["total_pp"], 
    axis=1)
  pred_df["max_pp"] = np.maximum(pred_df["current_pp"], pred_df["pred_pp"])
  pred_df["unweighted_pp_gain"] = np.maximum(0, pred_df["pred_pp"] - pred_df["current_pp"])
  pred_df = apply_weight_curve(pred_df)

  pred_df = pred_df[PRED_FEATURES].sort_values(by="weighted_pp_gain", ascending=False)
  return pred_df