# app.py
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename

# ----------------- Flask app config -----------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# ----------------- Load model and classes -----------------
model = load_model('plant_leaf_disease_cnn_model.h5')
# Ensure proper Python list to avoid 'undefined'
class_names = np.load('class_names.npy', allow_pickle=True).tolist()

# ----------------- Helper function -----------------
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# ----------------- Routes -----------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    if file and allowed_file(file.filename):
        # Save uploaded file
        filename = secure_filename(file.filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # ----------------- Preprocess image -----------------
        img = image.load_img(filepath, target_size=(64, 64))  # same as training
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0).astype('float32')
        img_array /= 255.0  # IMPORTANT: same as training
        
        # ----------------- Make prediction -----------------
        predictions = model.predict(img_array)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0])) * 100  # convert to percentage
        
        # ----------------- Get top 3 predictions -----------------
        top3_indices = np.argsort(predictions[0])[-3:][::-1]
        top3_predictions = [
            {"class": class_names[i], "confidence": round(float(predictions[0][i]) * 100, 2)}
            for i in top3_indices
        ]
        
        # ----------------- Return JSON result -----------------
        result = {
            "predicted_class": class_names[predicted_class_idx],
            "confidence": round(confidence, 2),
            "top3": top3_predictions
        }
        
        return jsonify(result)
    
    return jsonify({'error': 'Invalid file type'})

# ----------------- Run Flask -----------------
if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
