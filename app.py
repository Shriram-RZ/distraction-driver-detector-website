from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
from predict import predict_image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'app/uploads/'

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
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
                
                # Predict the image
                label, probability = predict_image(filepath)
                
                response = {
                    'label': label,
                    'probability': float(probability),  # Convert to standard float
                    'image_url': filename
                }
                return jsonify(response)
            except Exception as e:
                print(f"Error in index route: {e}")
                return jsonify({'error': str(e)}), 500
    return render_template('index.html')

@app.route('/test')
def test():
    return "Server is running"

if __name__ == '__main__':
    app.run(debug=True)
