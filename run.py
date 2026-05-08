import subprocess
import sys
import os
import time

def start_backend():
    print("Starting FastAPI Backend...")
    env = os.environ.copy()
    backend_process = subprocess.Popen(
        [sys.executable, "api/main.py"],
        env=env,
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    return backend_process

def start_frontend():
    print("Starting Vite Frontend...")
    frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
    
    # Use shell=True for npm on Windows
    frontend_process = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=frontend_dir,
        shell=True
    )
    return frontend_process

if __name__ == "__main__":
    try:
        backend = start_backend()
        
        # Give backend a moment to start
        time.sleep(2)
        
        frontend = start_frontend()
        
        print("\n" + "="*50)
        print("HGAT Fake News Detection System is Running!")
        print("Frontend: http://localhost:5173")
        print("Backend API: http://localhost:8000")
        print("="*50 + "\n")
        
        # Keep main thread alive
        backend.wait()
        frontend.wait()
        
    except KeyboardInterrupt:
        print("\nShutting down services...")
        backend.terminate()
        frontend.terminate()
        sys.exit(0)
