from lxml import etree
from typing import List, Tuple, Dict
import os
from . import BaseAnnotator
from utilities import register_annotation

@register_annotation
class PascalVOC(BaseAnnotator):
    def __init__(self, overwrite_classes: Dict[str, str] = None):
        super().__init__()
        self.root = etree.Element('annotation')
        self.objects = list()
        self.overwrite_classes = overwrite_classes
        segmented = etree.SubElement(self.root, 'segmented')
        segmented.text = '1'
    
    def reset(self):
        self.root = etree.Element('annotation')
        self.objects = list()
        segmented = etree.SubElement(self.root, 'segmented')
        segmented.text = '1'
        
    def append_object(self, bounding_box, class_label: str):
        bb, ordered_mask_points = self._extract_from_bounding_box(bounding_box)
        object_element = etree.SubElement(self.root, 'object')
        bb_element = self._get_bb_subtree(bb)
        object_element.append(bb_element)
        object_id = etree.SubElement(object_element, 'id')
        object_id.text = str(len(self.objects)+1)
        name = etree.SubElement(object_element, 'name')
        if self.overwrite_classes and class_label in self.overwrite_classes:
            class_label = self.overwrite_classes[class_label]
        name.text = class_label
        etree.SubElement(object_element, 'partId')
        segmentation_node = self._get_segmentation_subtree(ordered_mask_points)
        object_element.append(segmentation_node)

    def _extract_from_bounding_box(self, bounding_box):
        ordered_mask_points = bounding_box.squeeze().tolist()
        
        xmin = min(point[0] for point in ordered_mask_points)
        ymin = min(point[1] for point in ordered_mask_points)
        xmax = max(point[0] for point in ordered_mask_points)
        ymax = max(point[1] for point in ordered_mask_points)
        
        bb = [xmin, ymin, xmax, ymax]
        
        return bb, ordered_mask_points
    
    def has_detection(self):
        return bool(len(self.root.findall('object')))
    
    def set_size(self, size: Tuple[int, int, int]):
        size_element = etree.SubElement(self.root, 'size')
        w, h, d = size
        depth = etree.SubElement(size_element, 'depth')
        depth.text = str(d)
        width = etree.SubElement(size_element, 'width')
        width.text = str(w)
        height = etree.SubElement(size_element, 'height')
        height.text = str(h)
    
    def write_xml(self, output_path: str, size: Tuple[int, int, int]):
        self.set_size(size)
        filename_element = etree.SubElement(self.root, 'filename')
        filename_element.text = os.path.basename(output_path)
        tree = etree.ElementTree(self.root)
        tree.write(output_path, pretty_print=True, xml_declaration=True, encoding='utf-8')
        
    def _get_segmentation_subtree(self, ordered_mask_points):
        segmentation = etree.Element('segmentation')
        for i, (x,y) in enumerate(ordered_mask_points):
            point_element = etree.SubElement(segmentation, 'point')
            index = etree.SubElement(point_element, 'index')
            index.text = str(i)
            x_element = etree.SubElement(point_element, 'x')
            x_element.text = str(x)
            y_element = etree.SubElement(point_element, 'y')
            y_element.text = str(y)
        return segmentation
    
    def _get_bb_subtree(self, bb: List[float]):
        assert len(bb) == 4
        bb_element = etree.Element('bndbox')
        for text, value in zip(['xmin', 'ymin', 'xmax', 'ymax'], bb):
            element = etree.SubElement(bb_element, text)
            element.text = str(value)
        return bb_element
    
    def __getstate__(self):
        # Convert the root element to a string for pickling
        state = self.__dict__.copy()
        state['root'] = etree.tostring(self.root)
        return state

    def __setstate__(self, state):
        # Convert the string back to an element after unpickling
        self.__dict__.update(state)
        self.root = etree.fromstring(state['root'])