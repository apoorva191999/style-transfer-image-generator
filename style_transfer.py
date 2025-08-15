"""
Neural Style Transfer implementation using TensorFlow and Keras.

- Downloads base and style images (from URL or local path)
- Allows user to search for art styles using Art Institute of Chicago API
- Performs style transfer using VGG19 pre-trained model
- Displays the original and stylized images

"""

# Uploading necessary packages and environment setup 
import sys
import keras
from keras import backend as K
import pandas as pd 
import numpy as np
import os
from keras.preprocessing.image import load_img, save_img, img_to_array
import matplotlib.pyplot as plt
from keras.applications import vgg19
from tensorflow.keras.applications.vgg19 import VGG19
from keras.models import Model
from scipy.optimize import fmin_l_bfgs_b
import requests
from PIL import Image
from io import BytesIO
import shutil
import tensorflow as tf
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing import image

# === Functions ===

def download_from_url(url, save_path):
    """
    Download an image from a URL or copy it from a local path.
    
    Args:
        url (str): URL or local file path to the image.
        save_path (str): Local filepath where the image will be saved.
    
    Returns:
        str: Path where the image is saved locally.
    """
    if url.startswith("http://") or url.startswith("https://"):
        # Download from the web
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        img.save(save_path)
        print(f"Saved {save_path} from web")
    else:
        # Treat as local file path and copy
        shutil.copy(url, save_path)
        print(f"Copied local file to {save_path}")
    return save_path


def search_and_choose_art(prompt):
    """
    Search the Art Institute of Chicago API for artworks matching a prompt,
    display the top 3 results, and ask the user to choose one for style transfer.
    
    Args:
        prompt (str): Search keyword for artworks.
    
    Returns:
        str or None: Local path to the chosen style image or None if no results.
    """
    api_url = "https://api.artic.edu/api/v1/artworks/search"
    params = {
        "q": prompt,
        "limit": 3,
        "fields": "id,title,image_id"
    }
    r = requests.get(api_url, params=params)
    data = r.json()

    if not data["data"]:
        print("No artworks found. Try another prompt.")
        return None

    artworks = data["data"]
    img_urls = []

    # Plot the artworks side-by-side for user selection
    fig, axs = plt.subplots(1, len(artworks), figsize=(15, 5))
    for i, art in enumerate(artworks):
        image_url = f"https://www.artic.edu/iiif/2/{art['image_id']}/full/843,/0/default.jpg"
        img_urls.append(image_url)

        img_data = requests.get(image_url)
        img = Image.open(BytesIO(img_data.content))

        axs[i].imshow(img)
        axs[i].axis('off')
        axs[i].set_title(f"{i+1}. {art['title']}", fontsize=10)

    plt.show()

    # Prompt user to select artwork by number
    choice = int(input(f"Enter the number of the artwork you want (1-{len(artworks)}): "))
    chosen_url = img_urls[choice-1]
    return download_from_url(chosen_url, "style_image.jpg")


# === Main interactive flow to get images ===

base_url = input("Enter the link for your base image: ")
base_image_path = download_from_url(base_url, "base_image.jpg")

prompt = input("Enter a style prompt: ")
style_image_path = search_and_choose_art(prompt)

print("Ready for style transfer!")
print(f"Base image: {base_image_path}")
print(f"Style image: {style_image_path}")


# === Image processing functions ===

def preprocess_image_instantiator(img_path, img_nrows, img_ncols):
    """
    Load an image, resize it, convert to array and preprocess for VGG19 model input.
    
    Args:
        img_path (str): Path to image file.
        img_nrows (int): Target number of rows (height).
        img_ncols (int): Target number of columns (width).
    
    Returns:
        np.array: Preprocessed image tensor ready for model input.
    """
    img = load_img(img_path, target_size=(img_nrows, img_ncols))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return preprocess_input(img)


def deprocess_image(x):
    """
    Convert a processed image tensor back into a displayable image.
    
    Args:
        x (np.array): Processed image tensor.
    
    Returns:
        np.array: Image array normalized between 0 and 1 for plotting.
    """
    x = x.reshape((x.shape[1], x.shape[2], 3))
    # Add back the mean pixel values subtracted in preprocess_input
    x[:, :, 0] += 103.939  
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # Convert from BGR to RGB color space
    x = x[:, :, ::-1]
    # Normalize pixels to [0,1]
    return np.clip(x / 255.0, 0, 1)


def gram_matrix(tensor):
    """
    Calculate the Gram matrix to capture style features by measuring correlations 
    between different filter responses.
    
    Args:
        tensor (tf.Tensor): 4D tensor with shape (1, height, width, channels).
    
    Returns:
        tf.Tensor: Gram matrix.
    """
    assert len(tensor.shape) == 4, "Tensor must be 4D"
    x = tensor[0]
    # Reshape to (height*width, channels)
    x = tf.reshape(x, (-1, x.shape[2])) 
    x = tf.transpose(x)  
    gram = tf.matmul(x, tf.transpose(x))
    return gram


def compute_loss(base_content, target, style_features, combination_features, style_weight=1e-2, content_weight=1e4):
    """
    Compute the total loss as a weighted sum of content loss and style loss.
    
    Args:
        base_content (tf.Tensor): Feature representation of base content image.
        target (tf.Tensor): Feature representation of generated image.
        style_features (list): List of style features from style image layers.
        combination_features (list): List of style features from generated image layers.
        style_weight (float): Weight for style loss.
        content_weight (float): Weight for content loss.
    
    Returns:
        tf.Tensor: Scalar total loss.
    """
    loss = tf.zeros(shape=())

    # Content loss: difference between content of base and generated images
    loss += content_weight * tf.reduce_mean(tf.square(base_content - target))

    # Style loss: difference in Gram matrices of style and generated images
    for sf, cf in zip(style_features, combination_features):
        loss += style_weight * tf.reduce_mean(tf.square(gram_matrix(sf) - gram_matrix(cf)))

    return loss


# === Main Style Transfer function ===

def Run_StyleTransfer(base_image_path, style_image_path, iterations=100):
    """
    Perform neural style transfer to combine content of base image with style of style image.
    
    Args:
        base_image_path (str): Filepath to the base image.
        style_image_path (str): Filepath to the style image.
        iterations (int): Number of optimization iterations.
    
    Returns:
        np.array: Stylized image array normalized between 0 and 1.
    """
    # Load and preprocess images to target size preserving aspect ratio
    width, height = load_img(base_image_path).size
    img_nrows = 400
    img_ncols = int(width * img_nrows / height)

    base_image = preprocess_image_instantiator(base_image_path, img_nrows, img_ncols)
    style_image = preprocess_image_instantiator(style_image_path, img_nrows, img_ncols)

    # Make the combination image a trainable variable initialized with the base image
    combination_image = tf.Variable(base_image, dtype=tf.float32)

    # Load pre-trained VGG19 model without top layers
    vgg = VGG19(include_top=False, weights='imagenet')
    # Extract features from specific convolutional layers relevant to style/content
    outputs = [vgg.get_layer(name).output for name in [
        'block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1', 'block5_conv2'
    ]]
    model = Model(inputs=vgg.input, outputs=outputs)
    model.trainable = False

    # Get style and content feature representations
    style_outputs = model(style_image)
    base_outputs = model(base_image)

    # First 5 outputs are style layers, last is content layer
    style_features = [layer for layer in style_outputs[:5]]
    base_content = base_outputs[-1]

    # Use Adam optimizer with a relatively high learning rate for faster convergence
    optimizer = tf.optimizers.Adam(learning_rate=5.0)

    @tf.function()
    def train_step():
        """
        Single optimization step to minimize style + content loss.
        """
        with tf.GradientTape() as tape:
            outputs = model(combination_image)
            combination_features = outputs[:5]
            combination_content = outputs[-1]
            loss = compute_loss(base_content, combination_content, style_features, combination_features)
        grad = tape.gradient(loss, combination_image)
        optimizer.apply_gradients([(grad, combination_image)])
        # Clip pixel values to keep them in a valid range
        combination_image.assign(tf.clip_by_value(combination_image, -128.0, 127.0))

    # Run optimization loop
    for i in range(iterations):
        train_step()

    final_img = combination_image.numpy()
    return deprocess_image(final_img)


# === Visualization of results ===

plt.figure(figsize=(30,30))

plt.subplot(5,5,1)
plt.title("Base Image", fontsize=20)
img_base = load_img(base_image_path)
plt.imshow(img_base)

plt.subplot(5,5,2)
plt.title("Style Image", fontsize=20)
img_style = load_img(style_image_path)
plt.imshow(img_style)

plt.subplot(5,5,3)
imgg = Run_StyleTransfer(base_image_path, style_image_path)
plt.title("Final Image", fontsize=20)
plt.imshow(imgg)

plt.tight_layout()
plt.show()
