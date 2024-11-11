import os
import random
from PIL import Image
from torchvision import transforms as transforms

def transform_image(img):
    transform = transforms.Compose([
        transforms.RandomChoice([
            transforms.RandomRotation(degrees=(0, 360)),
            transforms.RandomHorizontalFlip(p=1),
            transforms.RandomVerticalFlip(p=1)
        ]),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=(5, 9)),
        ], p=0.25)
    ])
    return transform(img)

def augment_images(folder_path, num_photos_needed):
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('jpg', 'jpeg', 'png'))]
    original_count = len(image_files)
    
    if original_count >= num_photos_needed:
        print("The folder already contains enough images.")
        return
    
    target_count = num_photos_needed - original_count
    saved_count = 0

    while saved_count < target_count:
        img_path = os.path.join(folder_path, random.choice(image_files))
        img = Image.open(img_path)
        
        transformed_img = transform_image(img)
        
        new_img_name = f"augmented_{saved_count + original_count}.jpg"
        transformed_img.save(os.path.join(folder_path, new_img_name))
        saved_count += 1

    print(f"Augmented {saved_count} images to reach the desired total of {num_photos_needed} images.")


folder_path = '../test_balance'
num_photos_needed = 10
augment_images(folder_path, num_photos_needed)
