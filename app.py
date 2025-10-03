import os
import numpy as np
from flask import Flask, request, render_template, jsonify, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

# Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Model load
MODEL_PATH = "plant_leaf_disease_cnn_model.h5"
model = load_model(MODEL_PATH)

# Class names (update these based on your actual classes)
class_names = ["Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", 
               "Apple___healthy", "Blueberry___healthy", "Cherry___Powdery_mildew", 
               "Cherry___healthy", "Corn___Cercospora_leaf_spot Gray_leaf_spot", 
               "Corn___Common_rust", "Corn___Northern_Leaf_Blight", "Corn___healthy", 
               "Grape___Black_rot", "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight", 
               "Grape___healthy", "Orange___Haunglongbing", "Peach___Bacterial_spot", 
               "Peach___healthy", "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", 
               "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy", 
               "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew", 
               "Strawberry___Leaf_scorch", "Strawberry___healthy", "Tomato___Bacterial_spot", 
               "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold", 
               "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite", 
               "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", 
               "Tomato___Tomato_mosaic_virus", "Tomato___healthy"]

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Serve uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"})

    if file and allowed_file(file.filename):
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(filepath)

        try:
            # Preprocess image
            img = image.load_img(filepath, target_size=(64, 64))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Prediction
            predictions = model.predict(img_array)
            predicted_class_idx = np.argmax(predictions[0])
            predicted_class = class_names[predicted_class_idx]
            confidence = float(np.max(predictions[0]))

            # Get top 3 predictions
            top3_indices = np.argsort(predictions[0])[-3:][::-1]
            top3_predictions = [
                {"class": class_names[i], "confidence": round(float(predictions[0][i]) * 100, 2)}
                for i in top3_indices
            ]

            return jsonify({
                "status": "success",
                "predicted_class": predicted_class,
                "confidence": round(confidence * 100, 2),
                "top3": top3_predictions,
                "image_url": f"/uploads/{filename}"
            })
        except Exception as e:
            return jsonify({"error": f"Error processing image: {str(e)}"})
    else:
        return jsonify({"error": "Invalid file type. Please upload an image file."})

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)