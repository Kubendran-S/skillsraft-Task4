# Task-4


# Install required packages
!pip install gradio opencv-python matplotlib numpy tensorflow scikit-learn

# Import necessary libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import gradio as gr
import urllib.request
import zipfile
from PIL import Image
import requests
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# Create directories for dataset
os.makedirs('dataset', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Download a reliable hand gesture dataset
print("Downloading hand gesture dataset...")
try:
    # Using a different hand gesture dataset that's more reliable
    dataset_url = "https://github.com/atif-hassan/Gesture_Dataset/raw/master/dataset.zip"
    urllib.request.urlretrieve(dataset_url, "dataset.zip")
    
    # Extract dataset
    with zipfile.ZipFile("dataset.zip", 'r') as zip_ref:
        zip_ref.extractall("dataset")
    
    dataset_path = "dataset"
    print("Dataset downloaded successfully!")
    
except Exception as e:
    print(f"Error downloading dataset: {e}")
    print("Creating synthetic dataset for demonstration...")
    dataset_path = "synthetic_dataset"
    os.makedirs(dataset_path, exist_ok=True)

# If dataset download failed or dataset is empty, create synthetic data
if not os.path.exists(dataset_path) or len(os.listdir(dataset_path)) == 0:
    def create_synthetic_dataset():
        print("Creating synthetic dataset for demonstration...")
        synthetic_path = "synthetic_dataset"
        os.makedirs(synthetic_path, exist_ok=True)
        
        # Create 5 gesture classes
        gestures = ['thumbs_up', 'peace', 'ok', 'fist', 'open_palm']
        
        for gesture_id, gesture_name in enumerate(gestures):
            gesture_dir = os.path.join(synthetic_path, gesture_name)
            os.makedirs(gesture_dir, exist_ok=True)
            
            # Create 100 synthetic images for each gesture
            for i in range(100):
                # Create a blank image with random background
                img = np.random.randint(50, 150, (64, 64, 3), dtype=np.uint8)
                
                # Draw different shapes based on gesture type
                if gesture_name == 'thumbs_up':
                    cv2.circle(img, (32, 20), 10, (255, 255, 255), -1)  # Head
                    cv2.rectangle(img, (27, 30), (37, 50), (255, 255, 255), -1)  # Body
                    cv2.rectangle(img, (40, 40), (45, 45), (255, 255, 255), -1)  # Thumb
                    
                elif gesture_name == 'peace':
                    cv2.circle(img, (32, 20), 10, (255, 255, 255), -1)  # Head
                    cv2.rectangle(img, (27, 30), (37, 50), (255, 255, 255), -1)  # Body
                    # Peace fingers
                    cv2.line(img, (37, 30), (45, 20), (255, 255, 255), 2)
                    cv2.line(img, (37, 30), (50, 30), (255, 255, 255), 2)
                    
                elif gesture_name == 'ok':
                    cv2.circle(img, (32, 32), 20, (255, 255, 255), 2)  # Circle
                    cv2.circle(img, (40, 25), 5, (255, 255, 255), -1)  # OK dot
                    
                elif gesture_name == 'fist':
                    cv2.circle(img, (32, 32), 20, (255, 255, 255), -1)  # Fist
                    
                elif gesture_name == 'open_palm':
                    cv2.circle(img, (32, 32), 15, (255, 255, 255), -1)  # Palm
                    # Fingers
                    for j in range(5):
                        cv2.line(img, (32, 17), (22 + j*5, 10), (255, 255, 255), 2)
                
                # Save the image
                cv2.imwrite(os.path.join(gesture_dir, f"{i}.jpg"), img)
        
        return synthetic_path, gestures
    
    dataset_path, gestures = create_synthetic_dataset()
else:
    # Get gesture classes from folder names
    gestures = sorted(os.listdir(dataset_path))
    gestures = [g for g in gestures if not g.startswith('.')]  # Remove hidden files
    # Limit to 5 classes for simplicity
    gestures = gestures[:5] if len(gestures) > 5 else gestures

print(f"Gesture classes: {gestures}")

# Load and preprocess the dataset
def load_dataset(path, img_size=(64, 64)):
    X = []
    y = []
    
    for label, gesture in enumerate(gestures):
        gesture_path = os.path.join(path, gesture)
        if not os.path.exists(gesture_path):
            continue
            
        image_files = [f for f in os.listdir(gesture_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_file in image_files[:100]:  # Limit to 100 images per class
            img_path = os.path.join(gesture_path, img_file)
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                    
                # Convert BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Resize and normalize image
                img = cv2.resize(img, img_size)
                img = img / 255.0
                
                X.append(img)
                y.append(label)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                continue
    
    return np.array(X), np.array(y)

# Load the dataset
X, y = load_dataset(dataset_path)
print(f"Dataset shape: {X.shape}")
print(f"Labels shape: {y.shape}")

# Check if we have enough data
if len(X) == 0:
    raise ValueError("No images were loaded. Please check your dataset path and structure.")

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

print(f"Training samples: {X_train.shape[0]}")
print(f"Validation samples: {X_val.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

# Define the CNN model
def create_model(num_classes, input_shape=(64, 64, 3)):
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# Create and train the model
num_classes = len(gestures)
model = create_model(num_classes)

# Check if we already have a trained model
model_path = "models/gesture_model.h5"
if os.path.exists(model_path):
    print("Loading pre-trained model...")
    model = keras.models.load_model(model_path)
else:
    print("Training the model...")
    # Add data augmentation to improve model generalization
    datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)
    
    datagen.fit(X_train)
    
    history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                        epochs=15, 
                        validation_data=(X_val, y_val),
                        verbose=1)
    
    # Save the model
    model.save(model_path)
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.4f}")

# Create a function for gesture prediction
def predict_gesture(image):
    """
    Predict the hand gesture from an image
    """
    # Preprocess the image
    image = cv2.resize(image, (64, 64))
    if len(image.shape) == 2:  # If grayscale, convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:  # If RGBA, convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    
    # Make prediction
    predictions = model.predict(image, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    
    return gestures[predicted_class], confidence

# Create Gradio interface
def gradio_predict(input_image):
    """
    Function for Gradio interface to predict gesture
    """
    # Convert from Gradio format to numpy array
    if isinstance(input_image, np.ndarray):
        image = input_image
    else:
        # Handle PIL Image or file path
        image = np.array(input_image)
    
    # Convert RGBA to RGB if needed
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # Predict gesture
    gesture, confidence = predict_gesture(image)
    
    # Create output text
    result = f"Predicted Gesture: {gesture}\nConfidence: {confidence:.2%}"
    
    return result

# Create examples for the Gradio interface
example_images = []
for gesture in gestures:
    gesture_path = os.path.join(dataset_path, gesture)
    if os.path.exists(gesture_path):
        img_files = [f for f in os.listdir(gesture_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        if img_files:
            example_images.append(os.path.join(gesture_path, img_files[0]))

# If no example images found, create some synthetic examples
if not example_images:
    for gesture in gestures:
        # Create a simple example image
        img = np.random.randint(50, 150, (64, 64, 3), dtype=np.uint8)
        
        if gesture == 'thumbs_up':
            cv2.circle(img, (32, 20), 10, (255, 255, 255), -1)
            cv2.rectangle(img, (27, 30), (37, 50), (255, 255, 255), -1)
            cv2.rectangle(img, (40, 40), (45, 45), (255, 255, 255), -1)
        elif gesture == 'peace':
            cv2.circle(img, (32, 20), 10, (255, 255, 255), -1)
            cv2.rectangle(img, (27, 30), (37, 50), (255, 255, 255), -1)
            cv2.line(img, (37, 30), (45, 20), (255, 255, 255), 2)
            cv2.line(img, (37, 30), (50, 30), (255, 255, 255), 2)
        
        example_path = f"example_{gesture}.jpg"
        cv2.imwrite(example_path, img)
        example_images.append(example_path)

# Create the Gradio interface
iface = gr.Interface(
    fn=gradio_predict,
    inputs=gr.Image(label="Upload Hand Gesture Image", type="numpy"),
    outputs=gr.Textbox(label="Prediction Result"),
    title="Hand Gesture Recognition System",
    description="Upload an image of a hand gesture, and the AI model will predict which gesture it is.",
    examples=example_images,
    allow_flagging="never"
)

# Launch the interface
iface.launch(debug=True, share=True)
