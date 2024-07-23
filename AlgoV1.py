import os
import certifi
import cv2
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog, Label, Button

# Ensure SSL certificates are set correctly
os.environ['SSL_CERT_FILE'] = certifi.where()

# Path to the folder containing images
image_folder = 'uploads'
image_size = (224, 224)  # Size required by pre-trained model

def load_images(image_folder):
    images = []
    filenames = []
    for filename in os.listdir(image_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img = cv2.imread(os.path.join(image_folder, filename))
            if img is not None:
                img = cv2.resize(img, image_size)
                images.append(img)
                filenames.append(filename)
    return np.array(images), filenames

# Load and preprocess images
images, filenames = load_images(image_folder)
preprocessed_images = preprocess_input(images)

# Load pre-trained model and extract features
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model = Model(inputs=base_model.input, outputs=base_model.output)
features = model.predict(preprocessed_images)

# Flatten the features and compute similarity
features_flattened = features.reshape(features.shape[0], -1)
similarity_matrix = cosine_similarity(features_flattened)

def find_similar_images(input_image_path, model, features_flattened, filenames, top_n=5):
    # Load and preprocess the input image
    input_image = cv2.imread(input_image_path)
    input_image_resized = cv2.resize(input_image, image_size)
    input_image_preprocessed = preprocess_input(np.expand_dims(input_image_resized, axis=0))

    # Extract features of the input image
    input_features = model.predict(input_image_preprocessed)
    input_features_flattened = input_features.reshape(1, -1)

    # Compute the cosine similarity between the input image and the dataset
    similarity_scores = cosine_similarity(input_features_flattened, features_flattened)
    similarity_scores = similarity_scores.flatten()

    # Get the indices of the most similar images
    similar_images_indices = similarity_scores.argsort()[-top_n-1:-1][::-1]
    similar_images = [filenames[i] for i in similar_images_indices]

    return input_image_path, similar_images

def display_images(input_image_path, similar_image_paths, title="Similar Images"):
    plt.figure(figsize=(15, 5))
    
    # Display the input image
    input_img = cv2.imread(input_image_path)
    input_img_rgb = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    plt.subplot(1, len(similar_image_paths) + 1, 1)
    plt.imshow(input_img_rgb)
    plt.title('Input Image')
    plt.axis('off')
    
    # Display the similar images
    for i, image_path in enumerate(similar_image_paths):
        img = cv2.imread(os.path.join(image_folder, image_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(1, len(similar_image_paths) + 1, i + 2)
        plt.imshow(img_rgb)
        plt.title(f'Similar Image {i+1}')
        plt.axis('off')
    
    plt.suptitle(title)
    plt.show()

def select_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        input_image_path, recommended_images = find_similar_images(file_path, model, features_flattened, filenames)
        print("Recommended images:", recommended_images)
        display_images(input_image_path, recommended_images)

# Create a simple GUI
root = Tk()
root.title("Image Recommender System")

label = Label(root, text="Select an image to find similar images")
label.pack(pady=10)

button = Button(root, text="Select Image", command=select_image)
button.pack(pady=10)

root.mainloop()
