from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle


def load_data():
    """Load the digits dataset (8x8 grayscale images)."""
    # TODO: Load the digits dataset
    # TODO: Extract images into X
    # TODO: Extract labels into y
    # TODO: Return X and y
    digits=load_digits()
    X,y=digits.images,digits.target
    return X,y


def flatten_images(X):
    """Flatten 2D (8x8) images into 1D feature vectors."""
    # TODO: Flatten images (n_samples, 8, 8) → (n_samples, 64)
    n_samples=X.shape[0]
    return X.reshape(n_samples,-1)


def split_data(X, y, test_size=0.2, random_state=42):
    """Split dataset into training and testing sets."""
    # TODO: Use train_test_split to split X and y
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test_size,random_state=random_state,shuffle=True)
    return X_train,X_test,y_train,y_test


def scale_features(X_train, X_test):
    """Scale features using StandardScaler."""
    # TODO: Create StandardScaler
    # TODO: Fit on training data
    # TODO: Transform both training and test data
    # TODO: Return scaled data and scaler
    scaler=StandardScaler()
    X_train_scaled=scaler.fit_transform(X_train)
    X_test_scaled=scaler.fit_transform(X_test)
    return X_train_scaled,X_test_scaled,scaler


if __name__ == "__main__":
    # 1. Load dataset
    X, y = load_data()

    # 2. Flatten images
    X_flat = flatten_images(X)

    # 3. Split into train/test sets
    X_train, X_test, y_train, y_test = split_data(X_flat, y)

    # 4. Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # 5. Save scaled data and scaler for reuse
    with open("scaled_data.pkl", "wb") as f:
        pickle.dump({
            "X_train": X_train_scaled,
            "X_test": X_test_scaled,
            "y_train": y_train,
            "y_test": y_test,
            "scaler": scaler
        }, f)

    print("\nPreprocessing Summary:")
    print(f"Training data shape : {X_train_scaled.shape}")
    print(f"Testing data shape  : {X_test_scaled.shape}")
    print("Scaled data and scaler saved as 'scaled_data.pkl'")
