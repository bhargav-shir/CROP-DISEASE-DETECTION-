"""
Crop Disease Detection App
---------------------------
Upload a leaf photo -> CNN predicts disease -> shows confidence + treatment/prevention info.
)
Run locally or in Google Colab:
    pip install gradio tensorflow pillow numpy
    python app.py

In Colab, use demo.launch(share=True) to get a public link.
"""

import json
import numpy as np
import gradio as gr
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image

# ---------------------------------------------------------------------------
# 1. Config
# ---------------------------------------------------------------------------

MODEL_PATH = r"C:\Users\RATHVA SUNIL\Downloads\MILESONE02\archive (5)"  # Your trained model
 # Your trained model

CLASS_NAMES_PATH = r"C:\Users\RATHVA SUNIL\Downloads\MILESONE02\class_names.json"

DISEASE_INFO_PATH = r"C:\Users\RATHVA SUNIL\Downloads\MILESONE02\disease_info.json"
# lookup table: class_name -> info dict
IMG_SIZE = (224, 224)
LOW_CONFIDENCE_THRESHOLD = 70.0                 # percent

# ---------------------------------------------------------------------------
# 2. Load model + supporting data (runs once at startup)
# ---------------------------------------------------------------------------

model = load_model(MODEL_PATH)

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)                  # e.g. ["Tomato_Early_Blight", "Potato_Late_Blight", ...]

with open(DISEASE_INFO_PATH, "r", encoding="utf-8") as f:
    disease_info = json.load(f)


# ---------------------------------------------------------------------------
# 3. Helper: build the formatted markdown output
# ---------------------------------------------------------------------------

def format_result(predicted_class: str, confidence: float) -> str:
    info = disease_info.get(predicted_class)

    if info is None:
        return (
            f"### {predicted_class}\n"
            f"**Confidence:** {confidence:.1f}%\n\n"
            "No treatment information available for this class yet."
        )

    display_name = info.get("display_name", predicted_class)
    cause = info.get("cause", "Not available")
    symptoms = info.get("symptoms", "Not available")
    treatment = info.get("treatment", [])
    prevention = info.get("prevention", [])

    treatment_md = "\n".join(f"- {t}" for t in treatment) or "- No treatment info available"
    prevention_md = "\n".join(f"- {p}" for p in prevention) or "- No prevention info available"

    output = (
        f"### {display_name}\n"
        f"**Confidence:** {confidence:.1f}%\n\n"
        f"**Cause:** {cause}\n\n"
        f"**Symptoms:** {symptoms}\n\n"
        f"**Treatment**\n{treatment_md}\n\n"
        f"**Prevention**\n{prevention_md}\n"
    )

    if display_name.lower() == "healthy":
        output = (
            f"### Healthy plant\n"
            f"**Confidence:** {confidence:.1f}%\n\n"
            "No disease detected. Keep up your current care routine."
        )
    elif confidence < LOW_CONFIDENCE_THRESHOLD:
        output += (
            "\n---\n"
            "**Note:** confidence is low. Try retaking the photo in better lighting, "
            "closer to the leaf, or consult a local agricultural expert to confirm."
        )

    output += "\n\n*This tool is for guidance only. For severe or spreading cases, consult a local agricultural expert.*"
    return output


# ---------------------------------------------------------------------------
# 4. Prediction function (called by Gradio on every submit)
# ---------------------------------------------------------------------------

def predict(image: Image.Image):
    if image is None:
        return "Please upload a leaf photo first."

    img = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)                 # matches MobileNetV2 training preprocessing
    arr = np.expand_dims(arr, axis=0)

    preds = model.predict(arr, verbose=0)[0]     # shape: (num_classes,)
    top_idx = int(np.argmax(preds))
    predicted_class = class_names[top_idx]
    confidence = float(preds[top_idx]) * 100

    return format_result(predicted_class, confidence)


# ---------------------------------------------------------------------------
# 5. Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="Crop Disease Detector") as demo:
    gr.Markdown("## 🌿 Crop Disease Detector\nUpload a photo of a plant leaf to check for disease and get treatment advice.")

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Leaf photo")
            submit_btn = gr.Button("Analyze", variant="primary")

        with gr.Column(scale=1):
            result_output = gr.Markdown(label="Result")

    submit_btn.click(fn=predict, inputs=image_input, outputs=result_output)
    image_input.change(fn=predict, inputs=image_input, outputs=result_output)  # auto-run on upload too

    gr.Examples(
        examples=[],  # add local sample image paths here, e.g. ["samples/tomato_leaf.jpg"]
        inputs=image_input,
    )

if __name__ == "__main__":
    # In Colab: demo.launch(share=True)
    demo.launch()
