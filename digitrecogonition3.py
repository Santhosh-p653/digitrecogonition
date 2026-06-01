import pickle
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from PIL import Image
import os

def load_scaled_data(file_path):
    """
    Load preprocessed and scaled data along with the scaler from a pickle file.

    **Important**
    - The pickle file is expected to contain a DICTIONARY with the following keys:
        "X_train", "X_test", "y_train", "y_test", "scaler"
    - Do NOT try to unpack it directly into variables, instead access values by keys.

    Args:
        file_path (str): Path to the pickle file.

    Returns:
        tuple: X_train, X_test, y_train, y_test, scaler
    """
    # TODO: Load scaled_data.pkl and return X_train, X_test, y_train, y_test, scaler
    with open(file_path,"rb") as f:
        data=pickle.load(f)
    X_train=data["X_train"]
    X_test=data["X_test"]
    y_train=data["y_train"]
    y_test=data["y_test"]
    scaler=data["scaler"]    
    return X_train,X_test,y_train,y_test,scaler

def build_model():
    """
    Create and return an MLPClassifier with specified hyperparameters.

    Hyperparameters:.load(f)
    - Single hidden layer with 64 neurons
    - ReLU activation function
    =
    - Adam optimizer
    - Training iterations → 50
    - Early stopping enabled: Stop if validation score does not improve
    - n_iter_no_change: 10 → iterations to wait before stopping
    - random_state: 42 → for reproducibility
    """
    model=MLPClassifier(
        hidden_layer_sizes=(64,),
        activation='relu',
        solver='adam',
        max_iter=50,
        early_stopping=True,
        n_iter_no_change=10,
        random_state=42
        )

    # TODO: Initialize MLPClassifier with above hyperparameters
    return model


def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    """Train the model on the training data and evaluate on the test data."""
    # TODO: Fit the model on X_train, y_train
    # TODO: Predict on X_test and calculate accuracy
    # TODO: Print test accuracy and classification report
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    acc=accuracy_score(y_test,y_pred)
    print(f"Test Accuracy:{acc:4f}")
    print("\nClasssification Report")
    print(classification_report(y_test,y_pred))
    return model

def predict_digits_from_images(model, scaler, image_files):
    """
    Predict digits from a list of grayscale 8x8 images.

    Args:
        model: Trained ML model.
        scaler: Fitted StandardScaler.
        image_files (list): List of image filenames.

    Returns:orm(img_)
        dict: Mapping from image filename to predicted digit.
    """
    predictions = {}
    # TODO: For each image file:
    #   1. Load image using Image.open()
    #   2. Resize the image to (8 X 8)
    #   2. Convert to numpy array
    #   3. Flatten and scale
    #   4. Predict using model
    #   5. Store prediction in dictionarypred
    for img_file in image_files:
        img=Image.open(img_file)
        img=img.resize((8,8))
        img_array=np.array(img)
        img_flat=img_array.reshape(1,-1)
        img_scaled=scaler.transform(img_flat)
        pred=model.predict(img_scaled)[0]
        predictions[img_file]=pred

    
    return predictions
    
if __name__ == "__main__":
    # 1. Load preprocessed scaled data
    X_train, X_test, y_train, y_test, scaler = load_scaled_data("scaled_data.pkl")

    # 2. Build, train, and evaluate model
    model = build_model()
    trained_model = train_and_evaluate(model, X_train, y_train, X_test, y_test)

    # 3. Predict digits from images 0.png to 9.png
    image_files = [f"{i}.png" for i in range(7) if os.path.exists(f"{i}.png")]
    predictions = predict_digits_from_images(trained_model, scaler, image_files)

    print("\nPredictions for images:")
    for img_file, pred in predictions.items():
        print(f"{img_file} → {pred}")

    # 4. Save trained model
    with open("model.pkl", "wb") as f:
        pickle.dump(trained_model, f)
