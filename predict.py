from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from model import create_model  # Ensure this imports correctly
import numpy as np
import operator

def predict_image(img_path):
    class_labels = ['safe_driving', 'texting_right', 'talking_on_phone_right', 'texting_left', 'talking_on_phone_left',
                     'operating_radio', 'drinking', 'reaching_behind', 'doing_hair_makeup', 'talking_to_passanger']
    
    model = create_model()
    model.load_weights("app/model/_weights.h5")
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    target_size = (150, 150)

    image = load_img(img_path, target_size=target_size)
    image_arr = img_to_array(image)
    image_arr = np.expand_dims(image_arr, axis=0)
    image_arr /= 255

    predictions = model.predict(image_arr)
    decoded_predictions = dict(zip(class_labels, predictions[0].tolist()))  # Convert to list
    sorted_predictions = sorted(decoded_predictions.items(), key=operator.itemgetter(1), reverse=True)

    # Return the class with the maximum probability
    return sorted_predictions[0]  # The most probable label and its probability
