import cv2
from predict import load_prediction_model, class_labels  # Import updated functions
from keras.preprocessing.image import img_to_array
import numpy as np
import os
import json

def predict_video(video_path, output_folder):
    # Load the model
    model = load_prediction_model()

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Error opening video file")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object
    output_filename = 'output_' + os.path.basename(video_path)
    output_path = os.path.join(output_folder, output_filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' for .avi files
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    frame_predictions = []  # List to store predictions for each frame
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Resize frame for prediction
            img = cv2.resize(frame, (150, 150))
            img_arr = img_to_array(img)
            img_arr = np.expand_dims(img_arr, axis=0)
            img_arr /= 255

            # Predict the frame
            predictions = model.predict(img_arr)
            label_idx = np.argmax(predictions[0])
            label = class_labels[label_idx]
            probability = predictions[0][label_idx]

            # Overlay prediction on frame
            text = f"{label}: {probability*100:.2f}%"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 255, 0), 2, cv2.LINE_AA)

            # Write the frame with prediction
            out.write(frame)

            # Calculate timestamp in seconds
            timestamp = frame_count / fps

            # Save the prediction for the current frame
            frame_predictions.append({
                'timestamp': round(timestamp, 2),  # Round timestamp to 2 decimal places
                'label': label,
                'probability': float(probability)  # Ensure probability is a standard float
            })
            frame_count += 1
        else:
            break

    cap.release()
    out.release()

    # Save predictions to a JSON file
    predictions_file = os.path.join(output_folder, 'predictions.json')
    with open(predictions_file, 'w') as f:
        json.dump(frame_predictions, f)

    return output_filename, 'predictions.json'  # Return both output video and predictions JSON filenames
