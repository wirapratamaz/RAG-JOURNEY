import os
import sys
from fastapi import FastAPI
from dotenv import load_dotenv

# Ensure the correct PYTHONPATH
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_dir not in sys.path:
    sys.path.append(project_dir)

from api.api import router as api_router  # Import correctly

# Load environment variables
load_dotenv()

app = FastAPI()
app.include_router(api_router, prefix="/api")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main_api:app", host="0.0.0.0", port=8000, reload=True)