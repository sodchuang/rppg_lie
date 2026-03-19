import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
import pandas as pd
import os

# Define directories and output files
directory_path = "C:\\Users\\User\\Desktop\\rppg\\gif"
output_file = "output_data_trend3.xlsx"
output_images_directory = "C:\\Users\\User\\Desktop\\rppg\\images"

# Create the directory for output images if it doesn't exist
os.makedirs(output_images_directory, exist_ok=True)

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Lists to store data from all GIF files
time_stamps_all = []
heart_rate_list_all = []
r_values_all = []
g_values_all = []
b_values_all = []
filenames_all = []
slopes_all = []

# Iterate over all files in the directory
for filename in os.listdir(directory_path):
    if filename.endswith(".gif"):
        video_path = os.path.join(directory_path, filename)
        
        heart_rate_list = []
        r_values = []
        g_values = []
        b_values = []

        cap = cv2.VideoCapture(video_path)
        frame_count = 0

        if not cap.isOpened():
            print(f"Error: Could not open video {filename}.")
            continue

        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        time_interval = 1 / frame_rate
        print(f"Processing {filename} with frame rate: {frame_rate}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the frame
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces) > 0:
                # Assuming we use the first detected face
                (x, y, w, h) = faces[0]
                
                # Extract the lower half of the face ROI
                lower_half_roi = frame[y + h // 2:y + h, x:x + w]
                cv2.rectangle(frame, (x, y + h // 2), (x + w, y + h), (0, 255, 0), 2)  # Draw rectangle around ROI

                gray_roi = cv2.cvtColor(lower_half_roi, cv2.COLOR_BGR2GRAY)
                mean_intensity = np.mean(gray_roi)

                heart_rate_list.append(mean_intensity)

                # Extract RGB mean values
                mean_rgb = cv2.mean(lower_half_roi)[:3]
                r_values.append(mean_rgb[2])
                g_values.append(mean_rgb[1])
                b_values.append(mean_rgb[0])

        cap.release()
        
        # Adjust time_stamps to match heart_rate_list length
        if heart_rate_list:
            time_stamps = np.arange(len(heart_rate_list)) * time_interval

            # Append data from this GIF to the main lists
            time_stamps_all.extend(time_stamps)
            heart_rate_list_all.extend(heart_rate_list)
            r_values_all.extend(r_values)
            g_values_all.extend(g_values)
            b_values_all.extend(b_values)
            filenames_all.extend([filename] * len(time_stamps))

            # Calculate trendline slope
            if len(time_stamps) > 1:  # Ensure there are enough points to fit a line
                coeffs = np.polyfit(time_stamps, heart_rate_list, 1)  # Linear fit
                slope = coeffs[0]
                slopes_all.extend([slope] * len(time_stamps))
                
                # Plot the original GIF and the rPPG waveform
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

                # Display the original frame with ROI rectangle
                cap = cv2.VideoCapture(video_path)
                ret, frame = cap.read()
                if ret:
                    ax1.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    ax1.set_title(f"Original GIF with ROI: {filename}")
                    ax1.axis('off')
                cap.release()

                # Plotting the rPPG waveform with trend line
                ax2.plot(time_stamps, heart_rate_list, label="Mean Intensity (Heart Rate Signal)")
                trend_line = np.polyval(coeffs, time_stamps)
                ax2.plot(time_stamps, trend_line, color='red', linestyle='--', label="Trend Line")
                ax2.set_xlabel("Time (seconds)")
                ax2.set_ylabel("Mean Intensity")
                ax2.set_title("rPPG Waveform with Trend Line")
                ax2.legend()

                # Save the plot as a JPG image
                image_output_path = os.path.join(output_images_directory, f"{filename}_plot.jpg")
                plt.savefig(image_output_path)
                plt.show()

            # Print rPPG values
            print(f"rPPG values for {filename}:")
            for t, hr in zip(time_stamps, heart_rate_list):
                print(f"Time: {t:.2f}s, rPPG: {hr:.2f}")

# Create a DataFrame to save the data
if time_stamps_all and heart_rate_list_all:  # Ensure there is data to save
    data = {
        "Filename": filenames_all,
        "Time (seconds)": time_stamps_all,
        "Mean Intensity (Heart Rate Signal)": heart_rate_list_all,
        "R Value": r_values_all,
        "G Value": g_values_all,
        "B Value": b_values_all,
        "Trendline Slope": slopes_all,
    }

    df = pd.DataFrame(data)

    # Save the DataFrame to an Excel file
    df.to_excel(output_file, index=False)
    print(f"Data saved to {output_file}")
else:
    print("No data to save.")
