import os
import numpy as np
from flask import Flask, request, render_template, jsonify
import logging
from werkzeug.utils import secure_filename

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-123')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Model loading with better error handling
model = None
class_names = []

try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image
    
    MODEL_PATH = "plant_leaf_disease_cnn_model.h5"
    if os.path.exists(MODEL_PATH):
        logger.info("Loading TensorFlow model...")
        model = load_model(MODEL_PATH)
        logger.info("✅ Model loaded successfully!")
    else:
        logger.warning("Model file not found, running in demo mode")
        
except Exception as e:
    logger.error(f"TensorFlow loading failed: {e}")
    model = None

# Class names
class_names = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", 
    "Apple___healthy", "Blueberry___healthy", "Cherry___Powdery_mildew", 
    "Cherry___healthy", "Corn___Cercospora_leaf_spot", 
    "Corn___Common_rust", "Corn___Northern_Leaf_Blight", "Corn___healthy", 
    "Grape___Black_rot", "Grape___Esca", "Grape___Leaf_blight", 
    "Grape___healthy", "Orange___Haunglongbing", "Peach___Bacterial_spot", 
    "Peach___healthy", "Pepper_bell___Bacterial_spot", "Pepper_bell___healthy", 
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy", 
    "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew", 
    "Strawberry___Leaf_scorch", "Strawberry___healthy", "Tomato___Bacterial_spot", 
    "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold", 
    "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites", 
    "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", 
    "Tomato___Tomato_mosaic_virus", "Tomato___healthy"
]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Create uploads directory at startup
with app.app_context():
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/health')
def health_check():
    return jsonify({
        "status": "healthy", 
        "model_loaded": model is not None,
        "service": "Plant Disease Detection API"
    })

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            file.save(filepath)

            # Demo mode if model not loaded
            if model is None:
                return jsonify({
                    "status": "demo_mode",
                    "predicted_class": "Tomato___healthy",
                    "confidence": 85.5,
                    "top3": [
                        {"class": "Tomato___healthy", "confidence": 85.5},
                        {"class": "Tomato___Early_blight", "confidence": 10.2},
                        {"class": "Tomato___Late_blight", "confidence": 4.3}
                    ],
                    "message": "Running in demo mode - model not loaded"
                })

            # Preprocess and predict
            img = image.load_img(filepath, target_size=(64, 64))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            predictions = model.predict(img_array, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            predicted_class = class_names[predicted_class_idx]
            confidence = float(np.max(predictions[0]))

            top3_indices = np.argsort(predictions[0])[-3:][::-1]
            top3_predictions = [
                {"class": class_names[i], "confidence": round(float(predictions[0][i]) * 100, 2)}
                for i in top3_indices
            ]

            # Cleanup
            if os.path.exists(filepath):
                os.remove(filepath)

            return jsonify({
                "status": "success",
                "predicted_class": predicted_class,
                "confidence": round(confidence * 100, 2),
                "top3": top3_predictions
            })
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return jsonify({"error": f"Error processing image: {str(e)}"}), 500
    else:
        return jsonify({"error": "Invalid file type. Please upload PNG, JPG, or JPEG."}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
