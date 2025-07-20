from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
import numpy as np
import os
import tensorflow as tf

def focal_loss_fixed(y_true, y_pred, gamma=2., alpha=.25):
    y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    cross_entropy = -y_true * tf.math.log(y_pred)
    weight = alpha * tf.math.pow(1 - y_pred, gamma)
    loss = weight * cross_entropy
    return tf.reduce_sum(loss, axis=1)

CONFIDENCE_THRESHOLD = 60  # percentage

app = Flask(__name__)
MobileNetV2_model = load_model('model\\MobileNetV2_finetuned.h5')
EfficientNetB0_model = load_model(
    'model\\EfficientNetB0_finetuned.keras',
    custom_objects={'focal_loss_fixed': focal_loss_fixed}
)
ResNet50_model = load_model(r"model\\resnet50_final_model.h5")

classes = ['bacterial_spot', 'early_blight', 'healthy', 'late_blight', 'leaf_mold',
           'mosaic_virus', 'septoria_leaf_spot', 'spider_mite', 'target_spot', 'yellow_leaf__curl_Virus']

UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        img_file = request.files['image']
        if img_file:
            filename = secure_filename(img_file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            img_file.save(filepath)

            img = image.load_img(filepath, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            predictions = MobileNetV2_model.predict(img_array)
            predicted_class_index = np.argmax(predictions)
            confidence = float(np.max(predictions)) * 100


            predicted_class = classes[predicted_class_index]
            return render_template(
                'index.html',
                prediction=predicted_class,
                confidence=confidence,
                img_path=filepath
            )

    return render_template('index.html', prediction=None)

@app.route('/model2', methods=['GET', 'POST'])
def model2_predict():
    if request.method == 'POST':
        img_file = request.files['image']
        if img_file:
            filename = secure_filename(img_file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            img_file.save(filepath)

            img = image.load_img(filepath, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            predictions = EfficientNetB0_model.predict(img_array)
            predicted_class_index = np.argmax(predictions)
            confidence = float(np.max(predictions)) * 100

           

            predicted_class = classes[predicted_class_index]
            return render_template(
                'index.html',
                prediction=predicted_class,
                confidence=confidence,
                img_path=filepath
            )

    return render_template('index.html', prediction=None)

@app.route('/model3', methods=['GET', 'POST'])
def model3_predict():
    if request.method == 'POST':
        img_file = request.files['image']
        if img_file:
            filename = secure_filename(img_file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            img_file.save(filepath)

            img = image.load_img(filepath, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            predictions = ResNet50_model.predict(img_array)
            predicted_class_index = np.argmax(predictions)
            confidence = float(np.max(predictions)) * 100

            

            predicted_class = classes[predicted_class_index]
            return render_template(
                'index.html',
                prediction=predicted_class,
                confidence=confidence,
                img_path=filepath
            )

    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True) 