#!/usr/bin/env python
"""
StormGate Dashboard Runner
"""
import subprocess
import sys
import os

def run_streamlit():
    """Run Streamlit dashboard"""
    try:
        # Get current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Build command
        cmd = [
            sys.executable,  # Use current Python interpreter
            "-m", "streamlit", "run", 
            "simple_dashboard.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ]
        
        print("Starting StormGate Dashboard...")
        print(f"Command: {' '.join(cmd)}")
        print("Dashboard will be available at: http://localhost:8501")
        print("Press Ctrl+C to stop the server")
        
        # Run command
        subprocess.run(cmd, cwd=current_dir)
        
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    except Exception as e:
        print(f"Error running dashboard: {e}")

if __name__ == "__main__":
    run_streamlit()
