import numpy as np
import gradio as gr
from PIL import Image
from tensorflow import keras

MODEL_PATH = "model3.keras"

# CIFAR-10 class order used during training.
CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

model = keras.models.load_model(MODEL_PATH)


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize((32, 32))
    arr = np.array(image).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)


def predict(image: Image.Image) -> dict:
    x = preprocess_image(image)
    probs = model.predict(x, verbose=0)[0]
    return {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload an image"),
    outputs=gr.Label(num_top_classes=3, label="Top predictions"),
    title="CIFAR-10 Model 3 Classifier",
    description=(
        "Upload an image and get CIFAR-10 class probabilities. "
        "Input is resized to 32x32 and normalized with pixel/255.0."
    ),
)


if __name__ == "__main__":
    demo.launch()
