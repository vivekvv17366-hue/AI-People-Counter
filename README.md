# AI People Counter

A real-time computer vision application that detects, tracks, and counts people crossing a virtual line using YOLO, OpenCV, and Streamlit.

## Project Overview

The AI People Counter monitors people entering and exiting an area through a webcam. It tracks detected individuals, maintains entry and exit counts, estimates current occupancy, and records counting data for analysis.

## Features

* Real-time person detection using YOLO
* Person tracking with unique tracking IDs
* Entry and exit counting using a virtual line
* Current occupancy estimation
* Live webcam interface using Streamlit and WebRTC
* Automatic counting data logging to CSV
* Occupancy trend visualization
* Counting history dashboard

## Technologies Used

* Python
* YOLO (Ultralytics)
* OpenCV
* Streamlit
* Streamlit WebRTC
* Pandas
* Matplotlib

## Project Structure

```text
AI_People_Counter/
├── app.py
├── analyze_data.py
├── requirements.txt
├── .gitignore
├── README.md
└── people_count.csv
```

*Your actual folder may contain additional files.*

## Installation

1. Clone the repository:

   ```bash
   git clone YOUR_GITHUB_REPOSITORY_URL
   ```

2. Navigate to the project folder:

   ```bash
   cd AI_People_Counter
   ```

3. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   ```

   Windows PowerShell:

   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Run the application:

   ```bash
   streamlit run app.py
   ```

The YOLO model weights are loaded by the application and may be downloaded automatically on the first run.

## Applications

Potential applications include retail footfall monitoring, classroom occupancy estimation, office space monitoring, and visitor-flow analysis.

## Limitations

Counting accuracy can be affected by overlapping people, missed detections, tracking-ID changes, camera angle, and lighting. Current occupancy is an estimate based on detected line crossings.

## Author

AI/ML and Data Science Project Portfolio
