import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import json

# Load the trained model
model = tf.keras.models.load_model('plant_leaf_disease_cnn_model.h5')

# Load class names
class_names = np.load('class_names.npy', allow_pickle=True)
print(f"Loaded {len(class_names)} classes")

# Configuration
IMG_SIZE = (64, 64)

# Function to preprocess the image
def preprocess_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize(IMG_SIZE)
    img_array = np.array(img)
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to predict the class of an image
def predict_disease(img_path, confidence_threshold=0.5):
    try:
        # Check if file exists
        if not os.path.exists(img_path):
            return {
                'status': 'error',
                'message': f'File not found: {img_path}'
            }
        
        # Preprocess the image
        processed_img = preprocess_image(img_path)
        
        # Make prediction
        predictions = model.predict(processed_img, verbose=0)
        
        # Get all predictions
        predicted_class_idx = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        # Get top 3 predictions
        top3_indices = np.argsort(predictions[0])[-3:][::-1]
        top3_predictions = [(class_names[i], float(predictions[0][i])) for i in top3_indices]
        
        # Check confidence level
        if confidence < confidence_threshold:
            return {
                'status': 'low_confidence',
                'predicted_class': class_names[predicted_class_idx],
                'confidence': float(confidence),
                'top_predictions': top3_predictions,
                'message': f'Low confidence ({confidence:.2%}). Try a clearer image.'
            }
        else:
            return {
                'status': 'success',
                'predicted_class': class_names[predicted_class_idx],
                'confidence': float(confidence),
                'top_predictions': top3_predictions,
                'message': 'Confident prediction'
            }
            
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Error processing image: {str(e)}'
        }

# Function to display results
def display_results(img_path, prediction_result):
    img = Image.open(img_path)
    
    plt.figure(figsize=(12, 5))
    
    # Display image
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title('Input Image')
    plt.axis('off')
    
    # Display prediction
    plt.subplot(1, 2, 2)
    plt.axis('off')
    
    if prediction_result['status'] == 'error':
        plt.text(0.1, 0.5, f"ERROR\n{prediction_result['message']}", 
                fontsize=12, transform=plt.gca().transAxes)
    
    else:
        text = f"Predicted: {prediction_result['predicted_class']}\n"
        text += f"Confidence: {prediction_result['confidence']:.2%}\n"
        text += f"Status: {prediction_result['message']}\n\n"
        text += "Top 3 Predictions:\n"
        
        for i, (cls, conf) in enumerate(prediction_result['top_predictions'], 1):
            text += f"{i}. {cls}: {conf:.2%}\n"
        
        plt.text(0.1, 0.9, text, fontsize=10, transform=plt.gca().transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

# Function to save results to JSON
def save_results(prediction_result, filename='prediction_results.json'):
    with open(filename, 'w') as f:
        json.dump(prediction_result, f, indent=2)
    print(f"Results saved to {filename}")

# Main function
def main():
    # Your test image path
    img_path = r"C:\Users\nimit\Music\.vscode\machine_leaning\Apple-black-rot-figure-1-shows-apple-black-rot-while-preprocessing-an-infected-leaf.png"
    print(f"Processing image: {img_path}")
    
    # Make prediction
    prediction_result = predict_disease(img_path)
    
    # Print results
    print("\n" + "="*50)
    print("PREDICTION RESULTS")
    print("="*50)
    
    if prediction_result['status'] == 'error':
        print(f"Error: {prediction_result['message']}")
    else:
        print(f"Predicted: {prediction_result['predicted_class']}")
        print(f"Confidence: {prediction_result['confidence']:.2%}")
        print(f"Status: {prediction_result['message']}")
        
        print("\nTop Predictions:")
        for i, (cls, conf) in enumerate(prediction_result['top_predictions'], 1):
            print(f"{i}. {cls}: {conf:.2%}")
    
    print("="*50)
    
    # Save results
    save_results(prediction_result)
    
    # Display visual results
    if prediction_result['status'] != 'error':
        display_results(img_path, prediction_result)
    
    return prediction_result

# Run prediction
if __name__ == "__main__":
    main()