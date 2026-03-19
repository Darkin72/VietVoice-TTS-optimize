"""
Main entry point for running vietvoicetts as a module with python -m vietvoicetts
"""

import uvicorn
from .server import app

def main():
    """Run the optimized FastAPI server"""
    print("Starting VietVoice-TTS Optimized Server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

if __name__ == "__main__":
    main()
