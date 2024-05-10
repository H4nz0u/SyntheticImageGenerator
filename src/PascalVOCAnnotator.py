from lxml import etree
from typing import Tuple, List, Dict


class PascalVOCAnnotation(object):
    def __init__(self, filename: str, original_path: str, size: Tuple[int, int, int]):
        self.root = etree.Element("annotation")
        self.objects = list()
        filename_element = etree.SubElement(self.root, "filename")
        filename_element.text = filename
        original_path_element = etree.SubElement(self.root, "original_path")
        original_path_element.text = original_path
        size_element = etree.SubElement(self.root, "size")
        h, w, c = size
        depth = etree.SubElement(size_element, "depth")
        depth.text = str(c)
        height = etree.SubElement(size_element, "height")
        height.text = str(h)
        width = etree.SubElement(size_element, "width")
        width.text = str(w)
        segmented = etree.SubElement(self.root, "segmented")
        segmented.text = "1"

    def append_object(
        self,
        bounding_box,
        class_label: str,
    ):
        bb, ordered_mask_points = self._extract_from_bb(bounding_box)
        obj_node = etree.SubElement(self.root, "object")
        bb_node = self._get_bb_subtree(bb)
        obj_node.append(bb_node)
        etree.SubElement(obj_node, "cadmodelname")
        obj_id = etree.SubElement(obj_node, "id")
        obj_id.text = str(len(self.objects) + 1)
        name = etree.SubElement(obj_node, "name")
        name.text = class_label
        etree.SubElement(obj_node, "partId")
        segmentation_node = self._get_segmentation_subtree(ordered_mask_points)
        obj_node.append(segmentation_node)

    def _get_bb_subtree(self, bb: List[float]):
        assert len(bb) == 4
        bndbox_obj = etree.Element("bndbox")
        for text, node_name in zip(bb, ["xmin", "ymin", "xmax", "ymax"]):
            sub = etree.SubElement(bndbox_obj, node_name)
            sub.text = str(text)
        return bndbox_obj

    def _get_segmentation_subtree(self, ordered_mask_points: List[Tuple[float, float]]):
        segmentation = etree.Element("segmentation")
        for i, (x, y) in enumerate(ordered_mask_points):
            point = etree.SubElement(segmentation, "point")
            index = etree.SubElement(point, "index")
            index.text = str(i)
            _x = etree.SubElement(point, "x")
            _x.text = str(x)
            _y = etree.SubElement(point, "y")
            _y.text = str(y)
        return segmentation

    def write_xml(self, output_path: str):
        tree = etree.ElementTree(self.root)
        tree.write(output_path, xml_declaration=True, encoding="UTF-8")

    def has_detection(self):
        return bool(len(self.root.findall("object")))
    
    def _extract_from_bb(self, bounding_box):
        # Convert the bounding box into a list of lists
        ordered_mask_points = bounding_box.squeeze().tolist()

        # Find the min and max coordinates to define the bounding box
        xmin = min(point[0] for point in ordered_mask_points)
        ymin = min(point[1] for point in ordered_mask_points)
        xmax = max(point[0] for point in ordered_mask_points)
        ymax = max(point[1] for point in ordered_mask_points)

        # Construct the bb dictionary
        bb = [xmin, ymin, xmax, ymax]

        return bb, ordered_mask_points
