import os
import sys

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath("src"))

# Import the main function from src/main.py
from src.main import main

# Call the main function
if __name__ == "__main__":
    main() 