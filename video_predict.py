import cv2
from predict import load_prediction_model, class_labels  # Import updated functions
from keras.preprocessing.image import img_to_array
import numpy as np
import os

def predict_video(video_path, output_folder):
    # Load the model
    model = load_prediction_model()

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    output_filename = 'output_' + os.path.basename(video_path)
    output_path = os.path.join(output_folder, output_filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' for .avi files
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height))

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
        else:
            break

    cap.release()
    out.release()

    return output_filename  # Return the output video filename
