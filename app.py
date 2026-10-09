
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from ultralytics import YOLO
import cv2
import csv
import os
from datetime import datetime
from threading import Lock


# -----------------------------------------
# 1. CONFIGURATION
# -----------------------------------------

st.set_page_config(
    page_title="AI People Counter",
    page_icon="👥",
    layout="wide"
)

CSV_FILE = "people_count.csv"

# Load YOLO model
model = YOLO("yolo11n.pt")


# -----------------------------------------
# 2. CSV LOGGING FUNCTION
# -----------------------------------------

csv_lock = Lock()


def save_count(entered, exited, current_people):
    with csv_lock:
        file_exists = os.path.exists(CSV_FILE)

        with open(CSV_FILE, "a", newline="") as file:
            writer = csv.writer(file)

            if not file_exists or os.path.getsize(CSV_FILE) == 0:
                writer.writerow([
                    "Time",
                    "Entered",
                    "Exited",
                    "Current People"
                ])

            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                entered,
                exited,
                current_people
            ])


# -----------------------------------------
# 3. PEOPLE COUNTER
# -----------------------------------------

class PeopleCounter(VideoProcessorBase):

    def __init__(self):
        self.entered = 0
        self.exited = 0

        # Remember each person's last side
        self.last_side = {}

        # Save data every 5 seconds
        self.last_save_time = datetime.now()

    def recv(self, frame):

        # Convert WebRTC frame to OpenCV image
        img = frame.to_ndarray(format="bgr24")

        # Frame dimensions
        height, width = img.shape[:2]

        # Counting line and buffer zone
        line_y = height // 2
        line_offset = 10

        # Run YOLO tracking
        results = model.track(
            img,
            persist=True,
            verbose=False
        )

        result = results[0]

        # Draw counting line
        cv2.line(
            img,
            (0, line_y),
            (width, line_y),
            (255, 0, 0),
            2
        )

        # ---------------------------------
        # 4. DETECT AND COUNT PEOPLE
        # ---------------------------------

        for box in result.boxes:

            # Class ID 0 means person
            class_id = int(box.cls[0])

            if class_id != 0:
                continue

            # Bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Detection confidence
            confidence = float(box.conf[0])

            # Skip detections without a tracking ID
            if box.id is None:
                continue

            track_id = int(box.id[0])

            # Calculate center point
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Determine which side of the line
            if center_y < line_y - line_offset:
                current_side = "above"

            elif center_y > line_y + line_offset:
                current_side = "below"

            else:
                current_side = "middle"

            # Check line crossing
            if current_side != "middle":

                if track_id in self.last_side:

                    previous_side = self.last_side[track_id]

                    # Above to below = entered
                    if (
                        previous_side == "above"
                        and current_side == "below"
                    ):
                        self.entered += 1

                    # Below to above = exited
                    elif (
                        previous_side == "below"
                        and current_side == "above"
                    ):
                        self.exited += 1

                # Update the person's last known side
                self.last_side[track_id] = current_side

            # Draw bounding box
            cv2.rectangle(
                img,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Draw center point
            cv2.circle(
                img,
                (center_x, center_y),
                5,
                (0, 0, 255),
                -1
            )

            # Display tracking ID and confidence
            label = f"ID: {track_id} | {confidence:.2f}"

            cv2.putText(
                img,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        # ---------------------------------
        # 5. CALCULATE CURRENT OCCUPANCY
        # ---------------------------------

        current_people = self.entered - self.exited

        # ---------------------------------
        # 6. SAVE CSV DATA EVERY 5 SECONDS
        # ---------------------------------

        now = datetime.now()

        if (now - self.last_save_time).total_seconds() >= 5:

            save_count(
                self.entered,
                self.exited,
                current_people
            )

            self.last_save_time = now

        # ---------------------------------
        # 7. DRAW DASHBOARD PANEL
        # ---------------------------------

        cv2.rectangle(
            img,
            (0, 0),
            (320, 140),
            (40, 40, 40),
            -1
        )

        cv2.putText(
            img,
            "AI PEOPLE COUNTER",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            img,
            f"ENTERED: {self.entered}",
            (15, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        cv2.putText(
            img,
            f"EXITED: {self.exited}",
            (15, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

        cv2.putText(
            img,
            f"CURRENT: {current_people}",
            (15, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # Return processed webcam frame
        return frame.from_ndarray(
            img,
            format="bgr24"
        )


# -----------------------------------------
# 8. STREAMLIT WEB INTERFACE
# -----------------------------------------

st.title("AI People Counter")
st.write(
    "Real-time people detection, tracking "
    "and occupancy monitoring."
)

st.info(
    "Click START below and allow camera access "
    "to begin counting."
)

webrtc_streamer(
    key="people-counter",
    video_processor_factory=PeopleCounter,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True
)


# -----------------------------------------
# 9. DISPLAY SAVED DATA
# -----------------------------------------

st.subheader("Saved Counting Data")

if os.path.exists(CSV_FILE) and os.path.getsize(CSV_FILE) > 0:

    import pandas as pd

    df = pd.read_csv(CSV_FILE)

    if not df.empty:

        # Display latest saved counts
        latest = df.iloc[-1]

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "People Entered",
            int(latest["Entered"])
        )

        col2.metric(
            "People Exited",
            int(latest["Exited"])
        )

        col3.metric(
            "Current People",
            int(latest["Current People"])
        )

        # Occupancy chart
        st.subheader("Occupancy Over Time")

        df["Time"] = pd.to_datetime(df["Time"])

        st.line_chart(
            df.set_index("Time")["Current People"]
        )

        # Data table
        st.subheader("Counting History")

        st.dataframe(
            df,
            use_container_width=True
        )

    else:
        st.warning("The CSV file has no records yet.")

else:
    st.warning(
        "No counting data saved yet. "
        "Start the camera and let it run for at least 5 seconds."
    )