import pandas as pd
import asyncio
from datetime import datetime, timezone
from app.services.fetcher import fetch_maps

maps_df = pd.DataFrame()
maps_time = datetime.now(timezone.utc)

# fetches ranked maps from beatleader api and loads them into memory
async def cache_maps():
  global maps_df, maps_time
  try:
    print("[Maps] Fetching ranked maps...")
    maps_df = await fetch_maps()
    maps_time = datetime.now(timezone.utc)
    print("[Maps] Refresh complete!")
  except:
    print("[Maps] Failed to refresh ranked maps.")

# refresh ranked maps every 2 hours
async def refresh_maps():
  while True:
    await asyncio.sleep(2 * 60 * 60) # 2 hours
    await cache_maps()

def get_cached_maps():
  return maps_df

def get_last_map_refresh():
  return maps_time