from sklearn.datasets import load_digits
import numpy as np


def load_data():
    """Load the digits dataset into arrays."""
    # TODO: Load the digits dataset
    # TODO: Extract images into a variable X
    # TODO: Extract labels into a variable y
    # TODO: Return X and y
    digits=load_digits()
    X= digits.images
    y=digits.target
    return X,y


def explore_data(X, y):
    """
    Analyze the dataset: shapes, class distribution, and value range.
    
    Args:
        X (np.ndarray): Images array
        y (np.ndarray): Labels array
    
    Returns:
        dict: Dictionary containing dataset information. Keys should include:
            - 'num_samples': Total number of images in the dataset
            - 'image_shape': Shape of a single image (height, width)
            - 'num_classes': Total number of unique classes
            - 'class_distribution': Dictionary mapping class label to its count
            - 'pixel_range': Tuple containing (min_pixel_value, max_pixel_value)
            - 'has_missing_values': Boolean indicating if X or y contain NaNs
    """
    # Create an empty dictionary to store dataset information
    info = {}

    # TODO: Compute and store the total number of samples
    # TODO: Determine and store the shape of a single image
    # TODO: Determine and store the number of unique classes
    # TODO: Compute and store the count of each class label
    # TODO: Determine the minimum and maximum pixel values
    # TODO: Check if there are any missing values in X or y
    
    # TODO: Return the dictionary containing all the above information
    num_samples=X.shape[0]
    info['num_samples']=num_samples
    image_shape=X.shape[1:]
    info['image_shape']=image_shape
    classes=np.unique(y)
    info['num_classes']=len(classes)
    class_counts={}
    for cls in classes:
        count=np.sum(y==cls)
        class_counts[cls]=count
    info['class_distribution']=class_counts
    min_pixel=X.min()
    max_pixel=X.max()
    info['pixel_range']=(min_pixel,max_pixel)
    has_missing=np.isnan(X).any() or np.isnan(y).any()
    info['has_missing_values']=has_missing
    return info


if __name__ == "__main__":
    # Load dataset
    X, y = load_data()
    
    dataset_info = explore_data(X, y)
    
    # Print the dataset information
    print("\n=== Digits Dataset Information ===")
    print(f"Total samples       : {dataset_info['num_samples']}")
    print(f"Image shape         : {dataset_info['image_shape']}")
    print(f"Number of classes   : {dataset_info['num_classes']}")
    print(f"Pixel value range   : {dataset_info['pixel_range']}")
    print(f"Has missing values? : {dataset_info['has_missing_values']}")
    print("\nClass distribution:")
    for cls, count in dataset_info['class_distribution'].items():
        print(f"  Class {cls}: {count} samples")