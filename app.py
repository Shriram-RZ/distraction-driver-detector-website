from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
from predict import predict_image
from video_predict import predict_video

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'app/uploads/'

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                # Predict the image
                label, probability = predict_image(filepath)
                response = {
                    'label': label,
                    'probability': float(probability),  # Convert to standard float
                    'file_url': filename,
                    'file_type': 'image'
                }
            elif filename.lower().endswith(('.mp4', '.avi', '.mov', '.webm')):
                # Predict the video and save the output video
                output_filename = predict_video(filepath, app.config['UPLOAD_FOLDER'])
                response = {
                    'file_url': output_filename,
                    'file_type': 'video'
                }
            else:
                return jsonify({'error': 'Unsupported file type'}), 400

            return jsonify(response)
        except Exception as e:
            print(f"Error in upload route: {e}")
            return jsonify({'error': str(e)}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    # Serve the file from the upload directory
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/test')
def test():
    return "Server is running"

if __name__ == '__main__':
    app.run(debug=True)
