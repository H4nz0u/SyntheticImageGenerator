from .boundingbox import BoundingBox
import numpy as np
from lxml import etree
from typing import Dict, Any

def parse_voc_xml(file_path) -> Dict[str, Any]:
    tree = etree.parse(file_path, None)
    objects = tree.findall('.//object')
    found_objects = []
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
        found_objects.append({
            'class': class_name,
            'bbox': {'x': xmin, 'y': ymin, 'width': width, 'height': height},
            'segmentation': np.array(segmentation, dtype=np.int32).reshape((-1, 1, 2))
        })
    return found_objects[0] # Only one object per annotation file

if __name__ == "__main__":
    print(parse_voc_xml("/data/horse/ws/joka888b-syntheticImageGenerator/SyntheticImageGenerator/images/manually_cut_out_signs/Klimaschild-China/image7182.xml"))