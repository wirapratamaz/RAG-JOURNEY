FROM python:3.9-slim

WORKDIR /app

# Copy requirements - with fallback
COPY requirements.txt requirements.txt 2>/dev/null || echo "fastapi>=0.95.0\nuvicorn>=0.22.0\npython-dotenv>=1.0.0\nlangchain>=0.1.5\nlangchain-community>=0.0.15\nlangchain-openai>=0.0.5\nopenai>=1.10.0\nchromadb>=0.6.3\npydantic>=2.5.0\ntiktoken>=0.5.0\nrequests>=2.30.0" > requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source files - check both locations
COPY src/ . 2>/dev/null || echo "Source directory not found at src/"
COPY *.py . 2>/dev/null || echo "No Python files in root"
COPY si_faqs.txt . 2>/dev/null || echo "si_faqs.txt not found"

# Create .env file with all required variables
RUN echo "CHROMA_URL=https://chroma-production-0925.up.railway.app\nCHROMA_TOKEN=qkj1zn522ppyn02i8wk5athfz98a1f4i\nOPENAI_API_KEY=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\nPORT=8000" > .env

# Create necessary directories
RUN mkdir -p standalone_chroma_db

# Expose port
EXPOSE 8000

# Set environment variables - these will be overridden by Railway if provided
ENV CHROMA_URL="https://chroma-production-0925.up.railway.app" 
ENV CHROMA_TOKEN="qkj1zn522ppyn02i8wk5athfz98a1f4i"
ENV PORT=8000

# Command to run the API
CMD uvicorn standalone_api:app --host 0.0.0.0 --port $PORT 