import os
import numpy as np
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Model loading with error handling
try:
    MODEL_PATH = "plant_leaf_disease_cnn_model.h5"
    model = load_model(MODEL_PATH)
    logging.info("✅ Model loaded successfully!")
except Exception as e:
    logging.error(f"❌ Error loading model: {e}")
    model = None

# Class names
class_names = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", 
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
    "Tomato___Tomato_mosaic_virus", "Tomato___healthy"
]

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Health check route for Render
@app.route('/health')
def health_check():
    return jsonify({"status": "healthy", "model_loaded": model is not None})

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Please try again later."}), 500
    
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"})

    if file and allowed_file(file.filename):
        try:
            # Save uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            file.save(filepath)

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

            # Clean up uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)

            return jsonify({
                "status": "success",
                "predicted_class": predicted_class,
                "confidence": round(confidence * 100, 2),
                "top3": top3_predictions,
                "image_url": f"/uploads/{filename}"  # Note: In production, use cloud storage
            })
            
        except Exception as e:
            # Clean up on error
            if 'filepath' in locals() and os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({"error": f"Error processing image: {str(e)}"}), 500
    else:
        return jsonify({"error": "Invalid file type. Please upload PNG, JPG, or JPEG."})

# Create uploads directory on startup
@app.before_first_request
def create_upload_dir():
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
