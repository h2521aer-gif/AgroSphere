import os
import numpy as np
from flask import Flask, request, render_template_string
from PIL import Image
import tensorflow as tf
import requests
import markdown  # tambahan

app = Flask(__name__)

# =============== CONFIG ==================
TFLITE_MODEL_PATH = "plant_disease_model.tflite"
API_KEY = "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"
API_URL = "https://openrouter.ai/api/v1/chat/completions"

CLASS_NAMES = [
    "Apple Apple Scab",
    "Apple Black Rot",
    "Apple Cedar Apple Rust",
    "Apple Healthy",
    "Blueberry Healthy",
    "Cherry Including Sour Powdery Mildew",
    "Cherry Including Sour Healthy",
    "Corn Maize Cercospora Leaf Spot Gray Leaf Spot",
    "Corn Maize Common Rust",
    "Corn Maize Northern Leaf Blight",
    "Corn Maize Healthy",
    "Grape Black Rot",
    "Grape Esca Black Measles",
    "Grape Leaf Blight Isariopsis Leaf Spot",
    "Grape Healthy",
    "Orange Haunglongbing Citrus Greening",
    "Peach Bacterial Spot",
    "Peach Healthy",
    "Pepper Bell Bacterial Spot",
    "Pepper Bell Healthy",
    "Potato Early Blight",
    "Potato Late Blight",
    "Potato Healthy",
    "Raspberry Healthy",
    "Soybean Healthy",
    "Squash Powdery Mildew",
    "Strawberry Leaf Scorch",
    "Strawberry Healthy",
    "Tomato Bacterial Spot",
    "Tomato Early Blight",
    "Tomato Late Blight",
    "Tomato Leaf Mold",
    "Tomato Septoria Leaf Spot",
    "Tomato Spider Mites Two Spotted Spider Mite",
    "Tomato Target Spot",
    "Tomato Tomato Yellow Leaf Curl Virus",
    "Tomato Tomato Mosaic Virus",
    "Tomato Healthy"
]

# =============== LOAD TFLITE MODEL ==================
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict_image(img_path):
    img = Image.open(img_path).convert("RGB").resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])
    predicted_idx = np.argmax(preds[0])
    return CLASS_NAMES[predicted_idx]

def get_disease_info(disease_name):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "HTTP-Referer": "http://localhost:5000",
        "X-Title": "Plant Disease Detector",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "x-ai/grok-4-fast:free",
        "messages": [
            {"role": "system", "content": "Kamu adalah ahli patologi tanaman. Jawab dalam bahasa Indonesia, gunakan format Markdown agar rapi."},
            {"role": "user", "content": f"Jelaskan secara singkat tentang penyakit tanaman ini: {disease_name}. Sertakan penyebab, gejala, dan cara pengendalian/perawatannya."}
        ]
    }
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=20)
        if response.status_code == 200:
            j = response.json()
            md_text = j["choices"][0]["message"]["content"].strip()
            return markdown.markdown(md_text)  # convert ke HTML
        else:
            print("Error from API:", response.status_code, response.text[:500])
            return f"<p><i>Tidak ada informasi (HTTP {response.status_code}).</i></p>"
    except Exception as e:
        print("Exception during API call:", e)
        return "<p><i>Tidak ada informasi (API error).</i></p>"

# =============== HTML TEMPLATE ===============
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Plant Disease Detector</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f9f9f9; }
        h1 { color: green; }
        form { margin-bottom: 20px; padding: 20px; background: white; border-radius: 8px; box-shadow: 0 0 5px rgba(0,0,0,0.1); }
        input[type=file] { margin: 10px 0; }
        button { padding: 8px 15px; background: green; color: white; border: none; border-radius: 5px; cursor: pointer; }
        .result { padding: 15px; margin-top: 20px; background: white; border-radius: 8px; box-shadow: 0 0 5px rgba(0,0,0,0.1); }
    </style>
</head>
<body>
    <h1>🌱 Plant Disease Detector</h1>
    <form action="/" method="post" enctype="multipart/form-data">
        <input type="file" name="file" accept="image/*" required>
        <br>
        <button type="submit">Prediksi</button>
    </form>

    {% if prediction %}
        <div class="result">
            <h2>Hasil Prediksi: {{ prediction }}</h2>
            <div><b>Detail Penyakit:</b></div>
            <div>{{ details|safe }}</div>
        </div>
    {% endif %}
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "Tidak ada file yang diupload", 400
        file = request.files['file']
        if file.filename == '':
            return "Tidak ada file yang dipilih", 400

        file_path = os.path.join("uploads", file.filename)
        os.makedirs("uploads", exist_ok=True)
        file.save(file_path)

        prediction = predict_image(file_path)
        details = get_disease_info(prediction)

        return render_template_string(HTML_TEMPLATE, prediction=prediction, details=details)

    return render_template_string(HTML_TEMPLATE)

if __name__ == '__main__':
    app.run(debug=True)
