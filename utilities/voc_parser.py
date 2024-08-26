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
            'segmentation': np.array(segmentation, dtype=np.int32).reshape((-1, 1, 2))
        }

if __name__ == "__main__":
    print(parse_voc_xml("/data/horse/ws/joka888b-syntheticImageGenerator/SyntheticImageGenerator/images/manually_cut_out_signs/Klimaschild-China/image7182.xml"))