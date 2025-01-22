# Automatic Meter Reading with Raspberry Pi 5

This project enables real-time digit detection and meter reading using a Raspberry Pi 5, TensorFlow Lite, and YOLOv5s. It includes a web interface for displaying processed data.

---

## Setup Guide

### 1. Setting Up the Virtual Environment
1. Install `virtualenv` if not already installed:
   ```bash
   sudo apt install python3-virtualenv
   ```
2. Create a virtual environment:
   ```bash
   virtualenv ~/myenv
   ```
3. Activate the virtual environment:
   ```bash
   source ~/myenv/bin/activate
   ```

### 2. Installing Required Packages
1. Ensure you’re in the virtual environment (`myenv`) and install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### 3. Using Your Phone as the Camera
If you’re using a phone camera, follow these steps:
1. Install `jmtpfs`:
   ```bash
   sudo apt install jmtpfs
   ```
2. Mount your phone’s storage:
   ```bash
   jmtpfs ~/phone_storage
   ```
3. Update the `camera` directory in `main.py` to match your phone’s directory. For example:
   ```python
   camera_directory = "camera/Internal shared storage/DCIM/Camera"
   ```

### 4. Configuring Storage Mount and Script Paths
Update the following paths in `mount_and_run.sh` to match your setup:
- **Storage Path**: Directory where images are stored.
- **Main Script**: Path to the main Python script (`main.py`).
- **UI Script**: Path to the Flask app (`UI.py`).
- **Python Environment**: Python interpreter in your virtual environment.
- **Digits Directory**: Root directory for the project.
- **Log File**: Path to the log file for runtime logs.

Example `mount_and_run.sh`:
```bash
STORAGE_PATH=~/digits_detection/camera
MAIN_SCRIPT=~/digits_detection/main.py
UI_SCRIPT=~/digits_detection/UI.py
PYTHON_ENV=~/myenv/bin/python3
DIGITS_DIR=~/digits_detection
LOG_FILE=~/digits_detection/run_log.txt
```

### 5. Running the Project on Startup (Optional)
To automatically run the project on system startup:
1. Edit the crontab file:
   ```bash
   crontab -e
   ```
2. Add the following line to schedule the `mount_and_run.sh` script:
   ```bash
   @reboot /path/to/mount_and_run.sh
   ```
3. Save and exit the crontab.

Alternatively, you can run the script manually:
```bash
./mount_and_run.sh
```

---

## Usage
1. Connect all devices to the same Wi-Fi as the Raspberry Pi.
2. Access the web interface by navigating to the Raspberry Pi's IP address in a browser:
   ```
   http://<raspberry_pi_ip>:5000
   ```
3. View and analyze the processed meter readings.

---

## Notes
- Ensure the `VIDD` configuration allows proper storage mounting.
- Logs for runtime issues are available in `run_log.txt`.
- Modify paths in the scripts to fit your environment.

Feel free to explore the code and adapt it as needed! Contributions and feedback are welcome.
