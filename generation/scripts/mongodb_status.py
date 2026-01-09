"""
MongoDB Container Status

Check the status of the MongoDB container.
"""

import subprocess
import sys
from pathlib import Path


def check_mongodb_status():
    """Check MongoDB container status."""
    print("🔍 Checking MongoDB container status...\n")
    
    try:
        # Check if container exists
        result = subprocess.run(
            ['docker', 'ps', '-a', '--filter', 'name=am-qadf-mongodb', '--format', '{{.Names}}\t{{.Status}}\t{{.Ports}}'],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            output = result.stdout.strip()
            if output:
                print("📦 Container Information:")
                print("-" * 80)
                parts = output.split('\t')
                if len(parts) >= 2:
                    print(f"   Name: {parts[0]}")
                    print(f"   Status: {parts[1]}")
                    if len(parts) >= 3:
                        print(f"   Ports: {parts[2]}")
                print()
                
                # Check if running
                if 'Up' in output:
                    print("✅ MongoDB container is running")
                    
                    # Get detailed info
                    inspect_result = subprocess.run(
                        ['docker', 'inspect', 'am-qadf-mongodb', '--format', 
                         '{{.State.Health.Status}}'],
                        capture_output=True,
                        text=True
                    )
                    if inspect_result.returncode == 0 and inspect_result.stdout.strip():
                        health = inspect_result.stdout.strip()
                        print(f"   Health: {health}")
                    
                    return True
                else:
                    print("⚠️  MongoDB container exists but is not running")
                    print("   Start it with: python generation/scripts/start_mongodb.py")
                    return False
            else:
                print("❌ MongoDB container not found")
                print("   Start it with: python data_generation/scripts/start_mongodb.py")
                return False
        else:
            print(f"❌ Error checking container: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("❌ docker command not found. Please install Docker.")
        return False
    except Exception as e:
        print(f"❌ Error checking status: {e}")
        return False


if __name__ == "__main__":
    success = check_mongodb_status()
    sys.exit(0 if success else 1)




