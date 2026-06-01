
import numpy as np
import tensorflow as tf
import gradio as gr
from PIL import Image
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers, models

print("TensorFlow:", tf.__version__)
print("Gradio:", gr.__version__)

# ======================
# LOAD DATA
# ======================

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# ======================
# CNN MODEL
# ======================

model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),

    layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ======================
# TRAIN
# ======================

model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.1
)

test_loss, test_acc = model.evaluate(x_test, y_test)

print(f"\nTest Accuracy: {test_acc:.4f}")

# ======================
# PREDICTION FUNCTION
# ======================

def predict_digit(image):

    if image is None:
        return "Please draw a digit", {}

    try:

        # Newer Gradio versions sometimes return dict
        if isinstance(image, dict):
            image = image.get("composite", None)

        if image is None:
            return "Invalid image", {}

        image = np.array(image)

        # Convert to grayscale
        img = Image.fromarray(image).convert("L")

        # Resize to MNIST dimensions
        img = img.resize((28, 28))

        img = np.array(img)

        # Invert colors
        img = 255 - img

        # Normalize
        img = img.astype("float32") / 255.0

        # Reshape
        img = img.reshape(1, 28, 28, 1)

        prediction = model.predict(img, verbose=0)[0]

        digit = int(np.argmax(prediction))

        probabilities = {
            str(i): float(prediction[i])
            for i in range(10)
        }

        return f"Predicted Digit: {digit}", probabilities

    except Exception as e:
        return f"Error: {str(e)}", {}

# ======================
# GRADIO APP
# ======================

demo = gr.Interface(
    fn=predict_digit,

    inputs=gr.Sketchpad(
        height=300,
        width=300
    ),

    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Label(label="Confidence")
    ],

    title="Handwritten Digit Recognition",
    description="Draw a digit from 0-9 and click Submit."
)

demo.launch(
    share=True,
    debug=True
)