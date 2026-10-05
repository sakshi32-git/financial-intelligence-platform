"""
run_eye_control.py
==================
CLI entry point to launch and test Eye Control Mode directly from the terminal.

Usage:
    python run_eye_control.py
"""

import time
from src.eye_tracking.eye_tracker import eye_tracker

if __name__ == "__main__":
    print("==========================================")
    print("  👁️  Eye Control Mode Standalone Test   ")
    print("==========================================")
    print("Press Ctrl+C to stop.\n")

    res = eye_tracker.start()
    print("Start result:", res)

    try:
        while True:
            status = eye_tracker.get_status()
            print(f"\rStatus: {status['status_message']} | Cursor: {status['cursor_pos']} | EAR: {status['last_ear']}", end="", flush=True)
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping Eye Tracker...")
        eye_tracker.stop()
        print("Done.")
