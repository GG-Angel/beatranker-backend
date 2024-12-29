import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

def train_model(scores_df: pd.DataFrame) -> dict:
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

  return { model, X }