import numpy as np
import pandas as pd
from sklearn.base import r2_score
from sklearn.model_selection import train_test_split

def add_bias_column(X):
    """ Adds a y-intercept bias column to the given array.

    Args:
      X (arr): Can be 1-d or 2-d
    
    Returns:
      Xnew (arr): The same array, but 2-d with a column of 1's in the first spot
    """
    
    # If the array is 1-d
    if len(X.shape) == 1:
      Xnew = np.column_stack([np.ones(X.shape[0]), X])
    
    # If the array is 2-d
    elif len(X.shape) == 2:
      bias_col = np.ones((X.shape[0], 1))
      Xnew = np.hstack([bias_col, X])
        
    else:
      raise ValueError("Input array must be either 1-d or 2-d")

    return Xnew

def line_of_best_fit(X: np.array, y: np.array):
  """ Projects y onto the span of X to obtain the slope and y-intercept vector for the line of best fit.

  Params:
    X (1d-2d array): The array of predictor values
    y (1d array): The array of corresponding response values to X

  Returns:
    m (2x1 vector): The vector containing the coefficients for the line of best fit,
    where the 1st term is the y-intercept and the 2nd term is the slope.
  """

  # add the bias column to X
  X = add_bias_column(X)

  # use the formula to obtain the slope and y-intercept vector of the line of best fit
  XtXinv = np.linalg.inv(np.matmul(X.T, X))
  Xty = np.matmul(X.T, y)
  return np.matmul(XtXinv, Xty)

def linreg_predict(Xnew, ynew, m):
  """ Compute the residuals, mean squared error, and r2 between the true and predicted y-values with the line of best fit.

  Params:
    Xnew (1d-2d array): The array of predictor values
    ynew (1d array): The array of corresponding response values to Xnew
    m (2x1 vector): The vector containing the coefficients for the line of best fit from the line_of_best_fit() function

  Returns:
    result (dict): A dictionary containing information about the line of best fit.
    - ypreds: The predicted y-values from applying m to Xnew
    - resids: The residuals (differences) between the true and predicted y-values
    - mse: The mean squared error between ynew and ypreds
    - r2: The coefficient of determination representing the proportion of variability in ynew explained by the lobf
  """

  # compute predicted y-values based on dimensions of Xnew
  ypreds = (Xnew * m[1] + m[0]) if Xnew.ndim == 1 else (np.dot(Xnew, m[1:]) + m[0])

  # get residuals, mean squared errror, and r2
  resids = ynew - ypreds
  mse = (resids**2).mean()
  r2 = r2_score(ynew, ypreds) 

  return {
    "ypreds": ypreds,
    "resids": resids,
    "mse": mse,
    "r2": r2
  }

def train_improvement_model(scores_df: pd.DataFrame) -> np.array:
  """ Trains the model for improving existing scores using polynomial regression.

  Params:
    scores_df (df): The DataFrame of player scores

  Returns:
    model (arr): 1d array where the coefficients correspond to the rating and pp features
  """

  # set accuracy as dependent variable
  y = scores_df["accuracy"].to_numpy().reshape(-1, 1)

  # standardize X-features
  X_feats = scores_df[["passRating", "accRating", "techRating", "passPP", "accPP", "techPP"]]
  X_feats_scaled = (X_feats - X_feats.mean()) / X_feats.std()

  # generate transformed features
  X_played = np.column_stack([
      X_feats_scaled.to_numpy(),
      X_feats_scaled["passRating"] * X_feats_scaled["passPP"],
      X_feats_scaled["accRating"] * X_feats_scaled["accPP"],
      X_feats_scaled["techRating"] * X_feats_scaled["techPP"],
      np.exp(X_feats_scaled["passPP"]),
      np.exp(X_feats_scaled["accPP"]),
      np.exp(X_feats_scaled["techPP"])
  ])

  # train model
  model = line_of_best_fit(X_played, y)

  return model

def train_unplayed_model(scores_df: pd.DataFrame) -> np.array:
  """ Trains the model for potential scores on unplayed maps using exponential regression.

  Params:
    scores_df (df): The DataFrame of player scores

  Returns:
    model (arr): 1d array where the coefficients correspond to the inverted stars feature
  """

  # set accuracy as dependent variable and invert to mimic declining trend
  y_inv = (1 - scores_df["accuracy"]).to_numpy().reshape(-1, 1)
  y_inv_log = np.log(y_inv)

  # use stars as X-feature
  X_unplayed = scores_df["stars"].to_numpy().reshape(-1, 1)

  # train model
  model = line_of_best_fit(X_unplayed, y_inv_log)

  return model