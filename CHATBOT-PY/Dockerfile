FROM python:3.9-slim

WORKDIR /app

# Copy only necessary files
COPY requirements.txt .
COPY run.py .
COPY app ./app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create directory for ChromaDB (if using local fallback)
RUN mkdir -p ./chroma_db

# Expose port
EXPOSE 8000

# Command to run the API
CMD ["python", "run.py"]
