# Typing Time Data Collection
Track keyboard usage by detecting when hands rest over a (white) keyboard in a video feed. The script logs how long hands stay on the keyboard and writes the durations to a text file.

## What it does
- Detects a keyboard once per session and outlines it on the video.
- Tracks hand landmarks with MediaPipe; counts frames where hands are inside the keyboard contour.
- Logs each continuous hand-on-keyboard interval with timestamps (relative to the video) into `DATA_FILE.txt`.
- Displays a live countdown of remaining video time.

## Requirements
- Python 3.9+ (tested with CPython)
- OpenCV (cv2), MediaPipe, NumPy. Install with `pip install -r requirement.text`.

## Quick start (recorded video)
1) Create/activate an environment and install dependencies: `pip install -r requirement.text`.
2) Open `data_collection_typing_hand_tracking.py`.
3) Set the video path on the line: `cap = cv2.VideoCapture('YOUR_FILE_PATH')`.
4) Run the script. Press `q` to exit early. Results append to `DATA_FILE.txt` in the working directory.

## Real-time capture (webcam)
1) Swap the video capture line to the webcam version: `cap = cv2.VideoCapture(0)` and comment out the file-based line.
2) Run the script; ensure adequate lighting and a clearly visible white keyboard.

## Output format
Each line in `DATA_FILE.txt` looks like:  
`<seconds>; seconds at; <timestamp>; seconds; into the video`

## Tips
- Use consistent lighting so the keyboard contour is detected (threshold at value 200 in `detect_keyboard`).
- Adjust the keyboard color/threshold logic if using a non-white keyboard.
- Change the output file name by editing the `open("DATA_FILE.txt", ...)` call.

## Declaration of generative AI in coding
This script was originally drafted with help from OpenAI’s ChatGPT and then refined for this project.
