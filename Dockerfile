FROM python:3.9-slim

WORKDIR /app

# Copy requirements file first for better caching
COPY src/requirements.txt .

# Create requirements.txt if it doesn't exist yet
RUN if [ ! -f "requirements.txt" ]; then echo "Creating requirements.txt"; \
    echo "fastapi>=0.95.0\nuvicorn>=0.22.0\npython-dotenv>=1.0.0\nlangchain>=0.1.5\nlangchain-community>=0.0.15\nlangchain-openai>=0.0.5\nopenai>=1.10.0\nchromadb>=0.6.3\npydantic>=2.5.0\ntiktoken>=0.5.0" > requirements.txt; fi

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ .
COPY .env.railway ./.env

# Create directories
RUN mkdir -p ./standalone_chroma_db

# Expose the port
EXPOSE 8000

# Run the FastAPI app
CMD ["python", "standalone_api.py"] 