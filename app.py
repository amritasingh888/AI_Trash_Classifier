import gradio as gr
import cv2
import numpy as np
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("trash_classifier.h5")

classes = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

def predict_image(image):

    img = cv2.resize(image, (128, 128))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img, verbose=0)[0]

    result = {
        classes[i]: float(prediction[i])
        for i in range(len(classes))
    }

    predicted_class = classes[np.argmax(prediction)]
    confidence = float(np.max(prediction) * 100)

    summary = f"""
Prediction: {predicted_class}

Confidence: {confidence:.2f}%
"""

    return summary, result


with gr.Blocks(theme=gr.themes.Soft()) as demo:

    gr.Markdown(
        """
        # 🗑️ AI Trash Classifier
        
        Upload an image and classify waste into:
        
        - Cardboard
        - Glass
        - Metal
        - Paper
        - Plastic
        - Trash
        """
    )

    with gr.Row():

        input_image = gr.Image(
            type="numpy",
            label="Upload Waste Image"
        )

        output_text = gr.Textbox(
            label="Prediction"
        )

    output_label = gr.Label(
        label="Confidence Scores"
    )

    predict_btn = gr.Button("Classify Waste")

    predict_btn.click(
        fn=predict_image,
        inputs=input_image,
        outputs=[output_text, output_label]
    )

demo.launch()