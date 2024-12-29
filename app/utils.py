def filter_unplayed(scores_df, maps_df): 
  bool_unplayed = ~maps_df["leaderboardId"].isin(scores_df["leaderboardId"])
  return maps_df[bool_unplayed].copy()