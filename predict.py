from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from model import create_model  # Ensure this imports correctly
import numpy as np
import operator

# Define class labels globally so they can be imported
class_labels = ['safe_driving', 'texting_right', 'talking_on_phone_right', 
                'texting_left', 'talking_on_phone_left', 'operating_radio', 
                'drinking', 'reaching_behind', 'doing_hair_makeup', 
                'talking_to_passanger']

# Function to create/load model
def load_prediction_model():
    model = create_model()
    model.load_weights("app/model/_weights.h5")  # Ensure the path is correct
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model

# Predict for a single image
def predict_image(img_path):
    target_size = (150, 150)

    image = load_img(img_path, target_size=target_size)
    image_arr = img_to_array(image)
    image_arr = np.expand_dims(image_arr, axis=0)
    image_arr /= 255

    model = load_prediction_model()  # Load the model
    predictions = model.predict(image_arr)
    decoded_predictions = dict(zip(class_labels, predictions[0].tolist()))  # Convert to list
    sorted_predictions = sorted(decoded_predictions.items(), key=operator.itemgetter(1), reverse=True)

    # Return the class with the highest probability
    return sorted_predictions[0]  # The most probable label and its probability
