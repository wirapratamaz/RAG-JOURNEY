## Installation

1. **Clone the repository and navigate to the project directory:**
   ```bash
   git clone <repository-url>
   cd CHATBOT-PY
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv myvenv
   myvenv\Scripts\activate  # On Windows
   # source myvenv/bin/activate  # On macOS/Linux
   ```

3. **Install libraries and dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Get [OpenAI API key](https://platform.openai.com/account/api-keys) and set up environment variables:**
   - Create a `.env` file in the root directory.
   - Add the following:
     ```env
     OPENAI_API_KEY=your_openai_api_key
     CRAWL_URL=https://is.undiksha.ac.id/
     ```

5. **Install Playwright (if using asynchronous crawler):**
   ```bash
   playwright install chromium
   ```

## Running the Application

1. **Run the Streamlit app:**
   ```bash
   streamlit run src/main.py
   ```

2. **View the app in your browser at:**
   - Local URL: `http://localhost:8501`
   - Network URL: `http://<your-local-ip>:8501`

## Usage

The assistant will now utilize the enhanced Crawl4AI crawler to fetch and process information from the specified website, providing more accurate and comprehensive responses.

---