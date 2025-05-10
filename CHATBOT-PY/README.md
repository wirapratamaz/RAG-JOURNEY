# Undiksha RAG Chatbot API

A standalone API for the Undiksha RAG-based chatbot, designed for easy deployment.

## Features

- FastAPI-based REST API
- RAG (Retrieval-Augmented Generation) with OpenAI's GPT-3.5-turbo
- ChromaDB integration for vector storage
- Environment variable configuration
- Railway deployment support

## Project Structure

```
.
├── app/                 # Main application package
│   ├── api/             # API routes and models
│   │   ├── models.py    # Pydantic models
│   │   └── router.py    # API endpoints
│   └── core/            # Core business logic
│       └── rag_service.py # RAG implementation
├── Dockerfile           # Container definition
├── requirements.txt     # Python dependencies
├── railway.toml         # Railway configuration
├── run.py               # Application entry point
└── README.md            # This file
```

## Setup Instructions

### Local Development

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your configuration:
   ```
   OPENAI_API_KEY=your-openai-api-key
   CHROMA_URL=your-chroma-db-url
   CHROMA_TOKEN=your-chroma-db-token
   PORT=8000
   ```

5. Run the API:
   ```bash
   python run.py
   ```

The API will be available at `http://localhost:8000`.

### Deployment on Railway

1. Create a new Railway project.

2. Add the repository to your Railway project.

3. Set up the environment variables in the Railway dashboard:
   - `OPENAI_API_KEY`
   - `CHROMA_URL`
   - `CHROMA_TOKEN`

4. Deploy the application.

Railway will automatically build and deploy the API using the configuration in `railway.toml`.

## API Endpoints

- `GET /` - Welcome message
- `GET /health` - Health check endpoint
- `POST /api/query` - Process a query using the RAG system

### Query Example

```bash
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the internship procedure at Undiksha?"}'
```

Response:
```json
{
  "answer": "The answer to your query...",
  "sources": ["Source document content..."]
}
```