"""
Stop MongoDB Container

Stops the MongoDB container using docker-compose.
"""

import subprocess
import sys
from pathlib import Path


def stop_mongodb():
    """Stop MongoDB container using docker-compose."""
    # Get project root
    project_root = Path(__file__).parent.parent.parent
    docker_dir = project_root / 'docker'
    compose_file = docker_dir / 'docker-compose.dev.yml'
    
    if not compose_file.exists():
        print(f"❌ Docker compose file not found: {compose_file}")
        return False
    
    print("🛑 Stopping MongoDB container...")
    
    try:
        result = subprocess.run(
            ['docker-compose', '-f', str(compose_file), 'stop', 'mongodb'],
            cwd=str(docker_dir),
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✅ MongoDB container stopped successfully!")
            return True
        else:
            print(f"❌ Failed to stop MongoDB:")
            print(result.stderr)
            return False
            
    except FileNotFoundError:
        print("❌ docker-compose not found. Please install Docker Compose.")
        return False
    except Exception as e:
        print(f"❌ Error stopping MongoDB: {e}")
        return False


if __name__ == "__main__":
    success = stop_mongodb()
    sys.exit(0 if success else 1)




