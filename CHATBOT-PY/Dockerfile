FROM python:3.9-slim

WORKDIR /app

# Copy requirements - use the root directory requirements file
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and standalone API
COPY src/ src/
COPY standalone_api.py .
COPY si_faqs.txt .

# Create necessary directories
RUN mkdir -p standalone_chroma_db

# Expose port
EXPOSE 8000

# Set environment variables - these will be overridden by Railway
ENV CHROMA_URL="https://chroma-production-0925.up.railway.app" 
ENV CHROMA_TOKEN="qkj1zn522ppyn02i8wk5athfz98a1f4i"
ENV PORT=8000

# Command to run the API
CMD uvicorn standalone_api:app --host 0.0.0.0 --port $PORT 