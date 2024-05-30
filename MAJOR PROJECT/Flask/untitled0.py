import os
from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

app = Flask(__name__)
model = load_model('CNN.h5')

target_size = (256, 256)

@app.route('/')
def index():
    return render_template('main.html')

@app.route('/classification.html', methods=['GET', 'POST'])
def classification():
    if request.method == 'POST':
        f = request.files['image']
        filepath = os.path.join('uploads', f.filename)
        f.save(filepath)

        img = image.load_img(filepath, target_size=target_size)
        x = image.img_to_array(img)
        x = x / 255.0
        x = np.expand_dims(x, axis=0)

        prediction = np.argmax(model.predict(x), axis=1)
        labels = ['Full Water level', 'Half water level', 'Overflowing']
        predicted_label = labels[prediction[0]]

        return render_template('classification.html', prediction=predicted_label)

if __name__ == '__main__':
    app.run(debug=True)
