from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
#model = load_model('apple.h5')

model = tf.keras.models.load_model('apple.h5')

@app.route('/')
def index():
    return render_template('index.html')

def predict():
    # Assume you have a form with a file input named 'imagefile'
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    # Preprocess the image
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)

    # Make prediction
    yhat = model.predict(image)
    label = decode_predictions(yhat)
    label = label[0][0]
    classification = '%s (%.2f%%)' % (label[1], label[2] * 100)

    return jsonify({'prediction': classification})

if __name__ == '__main__':
    app.run(debug=True)

