import os
import numpy as np
from flask import Flask, request, render_template
from keras.models import load_model
from keras.preprocessing import image
from tensorflow.keras.utils import load_img, img_to_array

app = Flask(__name__, static_folder='static')
model = load_model('model.h5')

@app.route("/")
@app.route("/about")
def about():
    return render_template("about.html")
@app.route("/symptoms")
def symptoms():
    return render_template("symptoms.html")

@app.route("/check")
def check():
    return render_template("check.html")

@app.route("/predict",methods=["POST","GET"])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname('__file__')
        filepath = os.path.join(basepath, "uploads", f.filename)
        f.save(filepath)

        img = load_img(filepath, target_size=(64, 64))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)

        preds = model.predict(x)
        pred = np.argmax(preds, axis=1)
        print("prediction", pred)

        index = ['Left Bundle Branch Block', 'Normal', 'Premature Atrial Contraction','Premature Ventricular Contractions', 'Right Bundle Branch Block', 'Ventricular Fibrillation']
        result = str(index[pred[0]])
        return result
    return None



if __name__ == "__main__":
    app.run(debug=False)
