import os
import numpy as np
from PIL import Image, ImageOps
import random
from tqdm import tqdm  # Progress bar library
from concurrent.futures import ProcessPoolExecutor, as_completed
import psutil

# Define the paths
output_path = "Preprocessed"
base_path = "BraTS_Training"
classes = ['CT', 'IC', 'MP', 'NC', 'PN', 'WM']  # Ensure this matches your class list

if not os.path.exists(output_path):
    os.makedirs(output_path)
    for cls in classes:
        os.makedirs(os.path.join(output_path, cls))

def augment_image(img):
    # Define a list of augmentation techniques
    augmentations = [
        lambda x: x,  # Original
        lambda x: x.transpose(Image.FLIP_LEFT_RIGHT),  # Horizontal Flip
        lambda x: x.transpose(Image.FLIP_TOP_BOTTOM),  # Vertical Flip
        lambda x: x.rotate(90),  # Rotate 90 degrees
        lambda x: x.rotate(180),  # Rotate 180 degrees
        lambda x: x.rotate(270)  # Rotate 270 degrees
    ]
    augmented_images = [aug(img) for aug in augmentations]
    return augmented_images

def process_and_augment_image(image_path, target_size):
    img = Image.open(image_path)
    
    # Resize the image
    img = img.resize(target_size)
    
    # Augment the image
    augmented_images = augment_image(img)
    
    return augmented_images

def preprocess_class(cls, class_path, output_class_path, target_size, max_count):
    images = os.listdir(class_path)
    
    augmented_images = []
    
    # Limit the number of concurrent processes
    max_workers = min(8, psutil.cpu_count(logical=False))  # Use a fraction of available cores

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_and_augment_image, os.path.join(class_path, image_name), target_size) for image_name in images]
        
        for future in tqdm(as_completed(futures), desc=f"Processing {cls}", unit="image", total=len(images)):
            try:
                augmented_images.extend(future.result())
            except Exception as e:
                print(f"Error processing image: {e}")
    
    # Handle class imbalance by oversampling
    while len(augmented_images) < max_count:
        augmented_images.extend(random.sample(augmented_images, max_count - len(augmented_images)))
    
    # Ensure the augmented_images list has exactly max_count elements
    augmented_images = augmented_images[:max_count]
    
    # Convert to grayscale, normalize, and save the preprocessed images
    for i, img in tqdm(enumerate(augmented_images), desc=f"Saving {cls}", total=max_count, unit="image"):
        img = ImageOps.grayscale(img)
        img_array = np.array(img) / 255.0
        
        # Convert back to image
        img = Image.fromarray((img_array * 255).astype(np.uint8))
        
        # Save the preprocessed image
        img.save(os.path.join(output_class_path, f"{cls}_{i}.png"))

def preprocess_images(base_path, classes, output_path, target_size=(256, 256)):
    class_counts = {
        'CT': 34139,
        'IC': 14500,
        'MP': 4812,
        'NC': 29542,
        'PN': 9664,
        'WM': 3828
    }
    
    max_count = max(class_counts.values())
    max_class = max(class_counts, key=class_counts.get)
    
    for cls in classes:
        if cls == max_class:
            print(f"Skipping class {cls} as it has the highest number of samples.")
            continue
        
        class_path = os.path.join(base_path, cls)
        output_class_path = os.path.join(output_path, cls)
        
        preprocess_class(cls, class_path, output_class_path, target_size, max_count)

# Call the function to preprocess images
preprocess_images(base_path, classes, output_path, target_size=(256, 256))

print("Data preprocessing completed. Preprocessed images are saved to:", output_path)