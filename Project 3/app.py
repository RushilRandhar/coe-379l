import os
import numpy as np
import io
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
from flask import Flask, request, jsonify

app = Flask(__name__)

MODEL_PATH = 'best_hurricane_damage_model.h5'
model = load_model(MODEL_PATH)

model_summary = []
model.summary(print_fn=lambda x: model_summary.append(x))
model_summary_str = "\n".join(model_summary)

# preprocessing function
def preprocess_image(image, target_size=(128, 128)):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = image / 255.0  # Normalize to [0,1]
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/summary', methods=['GET'])
def get_summary():
    model_info = {
        "name": "Hurricane Harvey Damage Classifier",
        "type": model.__class__.__name__,
        "layers": len(model.layers),
        "input_shape": str(model.input_shape),
        "summary": model_summary_str,
        "total_params": model.count_params()
    }
    return jsonify(model_info)

@app.route('/inference', methods=['POST'])
def predict():
    try:
        if request.files and 'image' in request.files:
            image = Image.open(request.files['image'])
        elif 'file' in request.files:
            image = Image.open(request.files['file'])
        elif request.data:
            image_bytes = io.BytesIO(request.data)
            image = Image.open(image_bytes)
        else:
            return jsonify({"error": "No image provided"}), 400

        processed_image = preprocess_image(image)
        
        prediction = model.predict(processed_image)
        probability = float(prediction[0][0])
        
        if probability > 0.5:
            result = "damage"
        else:
            result = "no_damage"
        
        return jsonify({"prediction": result})
    
    except Exception as e:
        app.logger.error(f"Error processing request: {str(e)}")
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
