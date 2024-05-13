from .boundingbox import BoundingBox
import numpy as np
from lxml import etree


def parse_voc_xml(file_path):
    tree = etree.parse(file_path)
    objects = tree.findall('.//object')
    
    for obj in objects:
        # Parsing the bounding box
        bndbox = obj.find('bndbox')
        xmin = int(float(bndbox.find('xmin').text))
        ymin = int(float(bndbox.find('ymin').text))
        xmax = int(float(bndbox.find('xmax').text))
        ymax = int(float(bndbox.find('ymax').text))
        width = xmax - xmin
        height = ymax - ymin

        # Extracting the class name
        class_name = obj.find('name').text

        # Extracting the segmentation points
        segmentation = []
        points = obj.findall('.//segmentation/point')
        for point in points:
            x = int(float(point.find('x').text))
            y = int(float(point.find('y').text))
            segmentation.append((x, y))
        return {
            'class': class_name,
            'bbox': {'x': xmin, 'y': ymin, 'width': width, 'height': height},
            'segmentation': segmentation
        }

if __name__ == "__main__":
    print(parse_voc_xml("images\signs\image49478.xml"))