import os
import cv2
import xml.etree.ElementTree as ET
import numpy as np

# Path to the directory where images and annotations are stored
image_dir = 'output'
annotation_dir = 'output'
output_dir = 'visualized_annotations'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Function to parse XML and extract bounding box and segmentation points
def parse_annotation(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    objects = []
    for obj in root.findall('object'):
        bbox = obj.find('bndbox')
        segmentation = obj.find('segmentation')
        
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)

        points = []
        for point in segmentation.findall('point'):
            x = int(point.find('x').text)
            y = int(point.find('y').text)
            points.append((x, y))
        
        objects.append({
            'bbox': (xmin, ymin, xmax, ymax),
            'segmentation': points
        })
    
    return objects

# Iterate over the annotation files
for annotation_file in os.listdir(annotation_dir):
    if annotation_file.endswith('.xml'):
        xml_path = os.path.join(annotation_dir, annotation_file)
        objects = parse_annotation(xml_path)

        # Load the corresponding image
        image_file = annotation_file.replace('.xml', '.jpg')  # assuming images are .jpg
        image_path = os.path.join(image_dir, image_file)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Image {image_file} not found, skipping.")
            continue

        # Draw the bounding box and segmentation on the image
        for obj in objects:
            # Draw bounding box
            xmin, ymin, xmax, ymax = obj['bbox']
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            # Draw segmentation
            points = np.array(obj['segmentation'], np.int32)
            cv2.polylines(image, [points], isClosed=True, color=(0, 0, 255), thickness=2)

        # Save the annotated image
        output_path = os.path.join(output_dir, image_file)
        cv2.imwrite(output_path, image)

print("Annotations have been visualized and saved.")
