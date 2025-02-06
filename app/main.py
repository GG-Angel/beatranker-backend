import os
import asyncio
from fastapi.responses import JSONResponse
from fastapi import FastAPI, Request
from slowapi.errors import RateLimitExceeded
from app.routers import modifiers, recommendations
from app.services.maps import get_ranked_maps, refresh_maps
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address

app = FastAPI()
limiter = Limiter(key_func=get_remote_address)

app.state.limiter = limiter
app.add_middleware(
  CORSMiddleware,
  allow_origins=["https://www.beatranker.xyz", "https://beatranker.xyz"],
  allow_credentials=True,
  allow_methods=["GET", "PUT"],
  allow_headers=["*"],
)

app.include_router(recommendations.router)
app.include_router(modifiers.router)

# fetch ranked maps on startup and set task to refresh
@app.on_event("startup")
async def startup_event():
  await get_ranked_maps()
  asyncio.create_task(refresh_maps())

@app.exception_handler(RateLimitExceeded)
async def rate_limit_error(request: Request, exc: RateLimitExceeded):
  return JSONResponse(
    status_code=429,
    content={"detail": "Rate limit exceeded! Please try again later."}
  )

if __name__ == "__main__":
  import uvicorn
  uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))