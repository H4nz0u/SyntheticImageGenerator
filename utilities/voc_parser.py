import xml.etree.ElementTree as ET
from boundingbox import BoundingBox
def parse_voc_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    objects = []

    for obj in root.iter('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = BoundingBox(
            coordinates=(
                int(bbox.find('xmin').text),
                int(bbox.find('ymin').text),
                int(bbox.find('xmax').text) - int(bbox.find('xmin').text),
                int(bbox.find('ymax').text) - int(bbox.find('ymin').text)
            ),
            format_type="min_max"
        )
        objects.append(obj_struct)

    return objects