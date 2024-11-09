import os
from PIL import Image

def resize(input_folder, output_folder):
    valid_extensions = {'.jpg', '.jpeg', '.png'}
    os.makedirs(output_folder, exist_ok=True)
    dirs = os.listdir(input_folder)

    for item in dirs:
        input_path = os.path.join(input_folder, item)
        if os.path.isfile(input_path) and os.path.splitext(item)[1].lower() in valid_extensions:
            im = Image.open(input_path)
            im_resized = im.resize((200, 200))                 
            output_path = os.path.join(output_folder, os.path.splitext(item)[0] + '.jpg')
            im_resized.convert("RGB").save(output_path, 'JPEG')


directories = ['Biblis_hyperia_aganisa_images', 'Euptoieta_hegesia_meridiania_images', 
               'Morpho_helenor_images', 'Siproeta_stelenes_images']

for directory in directories:
    resize(f'../data/{directory}', f'../preprocessed_data/{directory}')
