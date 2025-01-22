#!/bin/bash
echo "Stopping gvfs processes..."
pkill gvfsd
pkill gvfs-gphoto2-volume-monitor
# Path variables
STORAGE_PATH=~/digits_detection/camera
MAIN_SCRIPT=~/digits_detection/main.py
UI_SCRIPT=~/digits_detection/UI.py
PYTHON_ENV=~/myenv/bin/python3
DIGITS_DIR=~/digits_detection
LOG_FILE=~/digits_detection/run_log.txt

MAX_RETRIES=5
RETRY_COUNT=0

# Logging setup
echo "Starting script at $(date)" > "$LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

# Function to stop conflicting services
stop_conflicting_services() {
    echo "Stopping conflicting services..."
    killall -q gvfsd-mtp gvfsd || true
}

# Function to mount the phone storage
mount_storage() {
    stop_conflicting_services
    echo "Attempting to mount phone storage..."
    while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
        jmtpfs -o allow_other "$STORAGE_PATH"
        if [ $? -eq 0 ]; then
            echo "Phone storage mounted successfully."
            break
        else
            echo "Failed to mount phone storage. Retrying in 10 seconds..."
            sleep 10
            RETRY_COUNT=$((RETRY_COUNT + 1))
        fi
    done

    if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
        echo "Failed to mount storage after $MAX_RETRIES attempts. Exiting."
        exit 1
    fi
}

# Function to unmount the phone storage
unmount_storage() {
    echo "Unmounting phone storage..."
    fusermount -u "$STORAGE_PATH"
}

# Function to run main.py with retry logic
run_main() {
    while true; do
        if mount | grep "$STORAGE_PATH" > /dev/null; then
            echo "Storage is already mounted."
        else
            mount_storage
        fi

        echo "Running main.py..."
        $PYTHON_ENV "$MAIN_SCRIPT"
        if [ $? -eq 0 ]; then
            echo "main.py completed successfully."
            break
        else
            echo "main.py encountered an error. Retrying after remounting storage..."
            unmount_storage
        fi
    done
}

# Function to start UI.py in the background
run_ui() {
    while true; do
        echo "Running UI.py..."
        $PYTHON_ENV "$UI_SCRIPT" &
        UI_PID=$!

        # Wait for UI.py to finish
        wait $UI_PID
        echo "UI.py stopped. Restarting in 5 seconds..."
        sleep 5
    done
}

# Cleanup on exit
trap 'echo "Stopping UI.py..."; kill $UI_PID; exit' SIGINT SIGTERM

# Activate virtual environment and change directory
cd "$DIGITS_DIR" || { echo "Failed to cd to $DIGITS_DIR"; exit 1; }
source ~/myenv/bin/activate || { echo "Failed to activate virtual environment"; exit 1; }

# Start UI.py in the background
run_ui &

# Run main.py with storage handling
run_main
