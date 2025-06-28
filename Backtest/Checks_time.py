import time
from datetime import datetime

try:
    while True:
        # Get the current time in YYYY-MM-DD HH:MM:SS format
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Print the time, return to the beginning of the line
        print(f"\r{current_time}", end='', flush=True)
        time.sleep(1)
except KeyboardInterrupt:
    # Allow the user to stop the script with Ctrl+C
    print("\nStopped")
