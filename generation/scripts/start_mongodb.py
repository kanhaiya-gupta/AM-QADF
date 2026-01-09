"""
Start MongoDB Container

Starts the MongoDB container using docker-compose.
Loads environment variables from development.env file.
"""

import subprocess
import sys
import os
from pathlib import Path
import time


def load_env_file(env_file: Path) -> dict:
    """Load environment variables from .env file."""
    env_vars = {}
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    # Remove quotes if present
                    value = value.strip('"\'')
                    env_vars[key] = value
    return env_vars


def start_mongodb():
    """Start MongoDB container using docker-compose."""
    # Get project root
    project_root = Path(__file__).parent.parent.parent
    docker_dir = project_root / 'docker'
    compose_file = docker_dir / 'docker-compose.dev.yml'
    env_file = project_root / 'development.env'
    
    if not compose_file.exists():
        print(f"❌ Docker compose file not found: {compose_file}")
        return False
    
    # Load environment variables from development.env
    env_vars = {}
    if env_file.exists():
        print(f"📄 Loading environment from: {env_file}")
        env_vars = load_env_file(env_file)
        # Set environment variables for subprocess
        for key, value in env_vars.items():
            os.environ[key] = value
        print(f"   Loaded {len(env_vars)} environment variables")
    else:
        print(f"⚠️  Environment file not found: {env_file}")
        print("   Using default values or system environment variables")
    
    print("🚀 Starting MongoDB container...")
    print(f"   Docker directory: {docker_dir}")
    print(f"   Compose file: {compose_file}")
    
    # Show MongoDB-related env vars
    mongo_vars = {k: v for k, v in env_vars.items() if 'MONGO' in k}
    if mongo_vars:
        print(f"   MongoDB config: {', '.join(mongo_vars.keys())}")
    
    try:
        # Start MongoDB service with environment variables
        env = os.environ.copy()
        result = subprocess.run(
            ['docker-compose', '-f', str(compose_file), 'up', '-d', 'mongodb'],
            cwd=str(docker_dir),
            env=env,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✅ MongoDB container started successfully!")
            print("\n⏳ Waiting for MongoDB to be ready...")
            time.sleep(5)
            
            # Check if container is running
            check_result = subprocess.run(
                ['docker', 'ps', '--filter', 'name=am-qadf-mongodb', '--format', '{{.Status}}'],
                capture_output=True,
                text=True
            )
            
            if check_result.returncode == 0 and check_result.stdout.strip():
                print(f"✅ MongoDB is running: {check_result.stdout.strip()}")
                print("\n📋 Next steps:")
                print("   - Check status: docker ps | grep mongodb")
                print("   - Check logs: docker logs am-qadf-mongodb")
                print("   - Test connection: python generation/scripts/check_mongodb.py")
                return True
            else:
                print("⚠️  Container started but status check failed")
                print("   Check logs: docker logs am-qadf-mongodb")
                return False
        else:
            print(f"❌ Failed to start MongoDB:")
            print(result.stderr)
            return False
            
    except FileNotFoundError:
        print("❌ docker-compose not found. Please install Docker Compose.")
        return False
    except Exception as e:
        print(f"❌ Error starting MongoDB: {e}")
        return False


if __name__ == "__main__":
    success = start_mongodb()
    sys.exit(0 if success else 1)

