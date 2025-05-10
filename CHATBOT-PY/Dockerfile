FROM python:3.9-slim

WORKDIR /app

# Create requirements.txt
RUN echo "fastapi>=0.95.0" > requirements.txt
RUN echo "uvicorn>=0.22.0" >> requirements.txt
RUN echo "python-dotenv>=1.0.0" >> requirements.txt
RUN echo "langchain>=0.1.5" >> requirements.txt
RUN echo "langchain-community>=0.0.15" >> requirements.txt
RUN echo "langchain-openai>=0.0.5" >> requirements.txt
RUN echo "openai>=1.10.0" >> requirements.txt
RUN echo "chromadb>=0.6.3" >> requirements.txt
RUN echo "pydantic>=2.5.0" >> requirements.txt
RUN echo "tiktoken>=0.5.0" >> requirements.txt
RUN echo "requests>=2.30.0" >> requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create standalone_api.py
RUN echo 'import os' > standalone_api.py
RUN echo 'from fastapi import FastAPI' >> standalone_api.py
RUN echo 'from fastapi.middleware.cors import CORSMiddleware' >> standalone_api.py
RUN echo '' >> standalone_api.py
RUN echo 'app = FastAPI()' >> standalone_api.py
RUN echo '' >> standalone_api.py
RUN echo 'app.add_middleware(' >> standalone_api.py
RUN echo '    CORSMiddleware,' >> standalone_api.py
RUN echo '    allow_origins=["*"],' >> standalone_api.py
RUN echo '    allow_credentials=True,' >> standalone_api.py
RUN echo '    allow_methods=["*"],' >> standalone_api.py
RUN echo '    allow_headers=["*"],' >> standalone_api.py
RUN echo ')' >> standalone_api.py
RUN echo '' >> standalone_api.py
RUN echo '@app.get("/health")' >> standalone_api.py
RUN echo 'async def health_check():' >> standalone_api.py
RUN echo '    return {"status": "healthy"}' >> standalone_api.py
RUN echo '' >> standalone_api.py
RUN echo '@app.get("/")' >> standalone_api.py
RUN echo 'async def root():' >> standalone_api.py
RUN echo '    return {"message": "Welcome to the API"}' >> standalone_api.py

# Set environment variables
ENV CHROMA_URL="https://chroma-production-0925.up.railway.app" 
ENV CHROMA_TOKEN="qkj1zn522ppyn02i8wk5athfz98a1f4i"
ENV PORT=8000

# Expose port
EXPOSE 8000

# Command to run the API
CMD ["uvicorn", "standalone_api:app", "--host", "0.0.0.0", "--port", "8000"] 