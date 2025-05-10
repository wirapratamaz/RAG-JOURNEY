FROM python:3.9-slim

WORKDIR /app

# Create requirements.txt directly in one command to reduce layers
RUN echo "fastapi>=0.95.0\nuvicorn>=0.22.0\npython-dotenv>=1.0.0\nlangchain>=0.1.5\nlangchain-community>=0.0.15\nlangchain-openai>=0.0.5\nopenai>=1.10.0\nchromadb>=0.6.3\npydantic>=2.5.0\ntiktoken>=0.5.0\nrequests>=2.30.0" > requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create standalone_api.py file
RUN echo 'import os\nfrom fastapi import FastAPI\nfrom fastapi.middleware.cors import CORSMiddleware\n\napp = FastAPI()\n\napp.add_middleware(\n    CORSMiddleware,\n    allow_origins=["*"],\n    allow_credentials=True,\n    allow_methods=["*"],\n    allow_headers=["*"],\n)\n\n@app.get("/health")\nasync def health_check():\n    return {"status": "healthy"}\n\n@app.get("/")\nasync def root():\n    return {"message": "Welcome to the API"}\n\nif __name__ == "__main__":\n    import uvicorn\n    port = int(os.getenv("PORT", "8000"))\n    uvicorn.run("standalone_api:app", host="0.0.0.0", port=port)' > standalone_api.py

# Create directory for ChromaDB
RUN mkdir -p standalone_chroma_db

# Expose port
EXPOSE 8000

# Command to run the API
CMD ["python", "standalone_api.py"]
