import os
import cv2
import xml.etree.ElementTree as ET
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Path to the directory where images and annotations are stored
image_dir = 'output'
annotation_dir = 'output'
output_dir = 'visualized_annotations'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

def parse_annotation(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    objects = []
    for obj in root.findall('object'):
        bbox = obj.find('bndbox')
        segmentation = obj.find('segmentation')
        
        if bbox is None:
            logger.warning(f"No <bndbox> found for an object in {xml_file}. Skipping this object.")
            continue
        
        try:
            xmin = int(float(bbox.find('xmin').text))
            ymin = int(float(bbox.find('ymin').text))
            xmax = int(float(bbox.find('xmax').text))
            ymax = int(float(bbox.find('ymax').text))
        except (AttributeError, ValueError) as e:
            logger.error(f"Error parsing bounding box in {xml_file}: {e}. Skipping this object.")
            continue

        points = []
        if segmentation is not None:
            for point in segmentation.findall('point'):
                try:
                    x = int(float(point.find('x').text))
                    y = int(float(point.find('y').text))
                    points.append((x, y))
                except (AttributeError, ValueError) as e:
                    logger.error(f"Error parsing segmentation point in {xml_file}: {e}. Skipping this point.")
                    continue
        else:
            logger.info(f"No <segmentation> found for an object in {xml_file}. Proceeding without segmentation.")

        objects.append({
            'bbox': (xmin, ymin, xmax, ymax),
            'segmentation': points
        })

    return objects


for annotation_file in os.listdir(annotation_dir):
    if annotation_file.endswith('.xml'):
        xml_path = os.path.join(annotation_dir, annotation_file)
        objects = parse_annotation(xml_path)

        base_name = os.path.splitext(annotation_file)[0]
        print(base_name)
        image_file = base_name + '.jpg'
        image_path = os.path.join(image_dir, image_file)
        print(image_path)
        if image_file is None:
            logger.warning(f"No corresponding image found for annotation {annotation_file}. Skipping.")
            continue

        image = cv2.imread(image_path)

        if image is None:
            logger.error(f"Failed to read image {image_file}. Skipping.")
            continue

        # Draw the bounding box and segmentation on the image
        for idx, obj in enumerate(objects, start=1):
            # Draw bounding box
            print(f"annotation_file {base_name}: {obj['bbox']}")
            xmin, ymin, xmax, ymax = obj['bbox']
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            # Optionally, add label text
            label = f"Object {idx}"
            cv2.putText(image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.9, (0, 255, 0), 2)

            # Draw segmentation
            points = np.array(obj['segmentation'], np.int32)
            if points.size > 0:
                # Ensure that there are enough points to form a polygon
                if len(points) >= 3:
                    cv2.polylines(image, [points], isClosed=True, color=(0, 0, 255), thickness=2)
                else:
                    logger.warning(f"Not enough segmentation points for object {idx} in {annotation_file}. Skipping segmentation drawing.")
            else:
                logger.info(f"No segmentation points for object {idx} in {annotation_file}. Skipping segmentation drawing.")

        # Save the annotated image
        output_path = os.path.join(output_dir, image_file)
        success = cv2.imwrite(output_path, image)
        if success:
            logger.info(f"Annotated image saved to {output_path}.")
        else:
            logger.error(f"Failed to save annotated image to {output_path}.")

print("Annotations have been visualized and saved.")
