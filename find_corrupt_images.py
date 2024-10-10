import os
from PIL import Image

def check_image(file_path):
    """
    Checks if a JPEG file is corrupted by verifying and fully loading the image.
    
    :param file_path: Path to the JPEG file to check.
    :return: True if the file is corrupted, False otherwise.
    """
    try:
        with Image.open(file_path) as img:
            img.verify()  # Verify the integrity of the file
            # Reload the image to reset the file pointer and detect further issues
        with Image.open(file_path) as img:
            img.load()  # Fully load the image
        return False  # No exception means the file is not corrupted
    except (IOError, SyntaxError, AttributeError) as e:
        print(f"Corrupted file detected: {file_path} - {e}")
        return True  # An exception indicates the file is corrupted

def check_images_in_directory(directory):
    """
    Recursively checks all JPEG files in the given directory for corruption.
    
    :param directory: Path to the directory to scan.
    """
    corrupted_files = []

    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg')):
                file_path = os.path.join(root, file)
                if check_image(file_path):
                    corrupted_files.append(file_path)

    if corrupted_files:
        print("\nCorrupted files found:")
        for file in corrupted_files:
            print(file)
    else:
        print("No corrupted JPEG files found.")

# Example usage
directory_path = '/data/horse/ws/joka888b-syntheticImageGenerator/SyntheticImageGenerator/images/backgrounds_dataset_GMD'
check_images_in_directory(directory_path)
