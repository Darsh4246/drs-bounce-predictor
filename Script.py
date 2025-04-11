import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
import streamlit as st
import tempfile

# ----------------- Streamlit Web Interface -----------------
st.set_page_config(layout="centered")
st.title("🏏 DRS Ball Tracker")

uploaded_file = st.file_uploader("Ay Samarth upload fast da", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    # ----------------- Core Logic -----------------
    ball_positions = []
    last_ball_position = None
    stump_box = None
    hit_stumps = False

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 360))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_green = np.array([30, 80, 80])
        upper_green = np.array([80, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)

        kernel = np.ones((3, 3), np.uint8)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_DILATE, kernel)

        ball_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in ball_contours:
            area = cv2.contourArea(cnt)
            if 30 < area < 300:
                (x, y, w, h) = cv2.boundingRect(cnt)
                aspect_ratio = float(w) / h
                if 0.6 < aspect_ratio < 1.4:
                    cx, cy = x + w // 2, y + h // 2
                    if cy > 100:
                        last_ball_position = (cx, cy)
                        ball_positions.append(last_ball_position)

        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([179, 255, 255])
        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = red_mask1 | red_mask2

        stump_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in stump_contours:
            area = cv2.contourArea(cnt)
            if area > 50:
                x, y, w, h = cv2.boundingRect(cnt)
                stump_box = (x, y, x + w, y + h)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 2, (255, 255, 0), -1)

        if len(ball_positions) >= 2:
            cv2.polylines(frame, [np.array(ball_positions)], False, (0, 255, 0), 2)

        if len(ball_positions) >= 5:
            xs, ys = zip(*ball_positions)
            coefficients = np.polyfit(xs, ys, 2)
            a, b, c = coefficients
            x_max = max(xs)
            x_future = np.linspace(x_max, x_max + 100, 30)
            y_future = a * x_future**2 + b * x_future + c
            future_path = np.array([(int(x), int(y)) for x, y in zip(x_future, y_future)])

            for x, y in future_path:
                cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)

            for x, y in future_path:
                if stump_box:
                    x1, y1, x2, y2 = stump_box
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        hit_stumps = True
                        break

            min_index = ys.index(min(ys))
            bounce_point = ball_positions[min_index]
            cv2.circle(frame, bounce_point, 8, (0, 0, 255), -1)

        if hit_stumps:
            cv2.putText(frame, "OUT!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        else:
            cv2.putText(frame, "Not Out", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()

    # Display results
    st.markdown("### 🖼️ Processed Video Frames")
    for f in frames[::3]:
        st.image(f)

else:
    st.info("Please upload a video to begin analysis.")
