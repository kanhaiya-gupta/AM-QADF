"""
AM-QADF Main Entry Point
Main application entry point that runs the client application
"""

import uvicorn
import logging
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Main entry point for the application."""
    logger.info("Starting AM-QADF Client Application...")
    
    # Import the app
    try:
        from client.app import app
    except ImportError as e:
        logger.error(f"Failed to import client app: {e}")
        logger.error("Make sure you're running from the project root directory")
        sys.exit(1)
    
    # Get configuration from environment or use defaults
    import os
    host = os.getenv("API_HOST", "0.0.0.0")  # Read from env or default to 0.0.0.0
    port = int(os.getenv("API_PORT", "8000"))  # Read from env or default to 8000
    reload = os.getenv("RELOAD", "true").lower() == "true"  # Enable auto-reload in development
    
    logger.info(f"Starting server on {host}:{port}")
    logger.info(f"Auto-reload: {reload}")
    
    # Run the application
    # Use uvicorn Config to ensure host binding works correctly
    # When reload=True, must use import string format
    app_path = "client.app:app" if reload else app
    
    config = uvicorn.Config(
        app_path,
        host=host,
        port=port,
        reload=reload,
        log_level="info",
        loop="asyncio"  # Explicitly set event loop
    )
    
    # Verify host configuration
    logger.info(f"Uvicorn config - Host: {config.host}, Port: {config.port}")
    
    server = uvicorn.Server(config)
    server.run()

if __name__ == "__main__":
    main()
