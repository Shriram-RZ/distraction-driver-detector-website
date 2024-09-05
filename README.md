pip install keras
pip install flask
pip install tensorflowBefore running the application, ensure you have the following installed:

- Python 3.x
- Flask
- Keras
- TensorFlow

You can install these packages by following the installation instructions below.

## Installation

To get started, follow these steps:

1. **Clone the repository:**

   If this project is hosted on GitHub, clone it using the following command:

   ```bash
   git clone <repository-url>
Then navigate to the project directory:

bash
Copy code
cd <repository-directory>
Install dependencies:

Use pip to install the required Python packages:

bash
Copy code
```
pip install keras
pip install flask
pip install tensorflow
```
These commands will install the Keras, Flask, and TensorFlow libraries.

Set up the environment (optional):

It is recommended to use a virtual environment to manage dependencies:

bash
Copy code
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Once the virtual environment is activated, install the dependencies again inside it.

Run the application:

To start the Flask server, use the following command:

bash
Copy code
flask run
The application will be running locally at http://127.0.0.1:5000/.

Usage
Open your browser and go to http://127.0.0.1:5000/.

You will be able to upload images or other relevant data through the web interface.

The application will process the input using the Keras and TensorFlow model and return a prediction.
TO RUN : flask run
