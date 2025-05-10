import os
import requests
import json
import argparse
import subprocess
import time
import signal
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Default API URL
API_URL = "http://localhost:8000"

# Test query
TEST_QUERY = "Apa saja pilihan tempat magang untuk mahasiswa Sistem Informasi?"

def start_api_server():
    """Start the API server in a subprocess"""
    print("Starting API server...")
    process = subprocess.Popen(
        ["python", "standalone_api.py"], 
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Give the server time to start
    time.sleep(5)
    
    return process

def stop_api_server(process):
    """Stop the API server"""
    print("Stopping API server...")
    process.terminate()
    process.wait()

def test_health_endpoint(url):
    """Test the health endpoint of the API"""
    try:
        response = requests.get(f"{url}/health")
        if response.status_code == 200:
            print("✅ Health check passed!")
            return True
        else:
            print(f"❌ Health check failed with status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_query_endpoint(url, query):
    """Test the query endpoint of the API"""
    try:
        data = {"query": query}
        response = requests.post(
            f"{url}/api/query", 
            json=data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Query endpoint passed!")
            print(f"\nQuery: {query}")
            print(f"\nAnswer: {result['answer']}")
            print(f"\nSources: {len(result['sources'])} sources found")
            return True
        else:
            print(f"❌ Query endpoint failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Query endpoint error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Test the RAG API")
    parser.add_argument("--url", type=str, default=API_URL, help="API URL to test (default: http://localhost:8000)")
    parser.add_argument("--query", type=str, default=TEST_QUERY, help="Query to test")
    parser.add_argument("--start-server", action="store_true", help="Start the API server before testing")
    args = parser.parse_args()
    
    process = None
    try:
        if args.start_server:
            process = start_api_server()
        
        # Test health endpoint
        health_ok = test_health_endpoint(args.url)
        
        if not health_ok:
            print("Health check failed, skipping query test")
            return 1
        
        # Test query endpoint
        query_ok = test_query_endpoint(args.url, args.query)
        
        if query_ok:
            print("\n✅ All tests passed!")
            return 0
        else:
            print("\n❌ Tests failed!")
            return 1
            
    finally:
        if process:
            stop_api_server(process)

if __name__ == "__main__":
    sys.exit(main()) 