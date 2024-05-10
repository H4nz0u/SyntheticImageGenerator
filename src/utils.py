import argparse
import os
import numpy as np
import random
import glob
from typing import Tuple
from lxml import etree
import yaml


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process sign and car images.")
    
    parser.add_argument("-s", "--signs", required=False, help="Path to sign images")
    parser.add_argument("-c", "--cars", required=False, help="Path to car images")
    parser.add_argument("-o", "--output", required=False, help="Output path for processed images")
    parser.add_argument("--config", required=False, help="Path to a config file")
    parser.add_argument("-bw", "--make_black_white", required=False, action="store_true",
                    help="Whether or not the generated images are black and white")
    parser.add_argument("-r", "--rotation", required=False, nargs=2, default=[0,0], type=int, metavar=("LOWER", "UPPER"), 
                        help="Bounds for rotation offset in degrees (e.g., --rotation -5 5).")
    parser.add_argument("-er", "--enable_rotation", required=False, action="store_true", help="If you want to enable rotation")
    parser.add_argument("-fr", "--fixed_rotation", required=False, default=0, type=int, help="Fixed rotation applied to image")
    parser.add_argument("-t", "--tilt_angle", required=False, nargs=2, default=[0, 0], type=float, metavar=("LOWER", "UPPER"), help="rotate the sign on the y axis")
    parser.add_argument("--size", required=False, nargs=2, default=[0.5, 0.5], type=float, metavar=("LOWER", "UPPER"),
                        help="Bounds for size of sign (e.g., --size 0.1 0.9).")
    parser.add_argument("-f", "--format", required=False,choices=["jpg", "jpeg", "png"], help="File Format for input images")
    parser.add_argument("-n", "--number", required=False, type=int, help="How many images should be generated?")
    parser.add_argument("-se", "--seed", required=False, default=random.randint(0, 2**64), type=int, help="Specify a seed to guarantee a reproduceable output")
    parser.add_argument("-p", "--position", required=False, default=0, type=float, help="Amount of variation in percent")
    parser.add_argument("--noise", required=False, action="store_true", help="Whether or not random noise should be added")
    parser.add_argument("--noise_intensity", required=False, default=0.02, type=float, help="Controls the intencity of the noise, should be between 0 and 0.1")
    parser.add_argument("--noise_correlation", required=False, default=300, type=float, help="Controls the correlation (patch size) of the noise, should be between 0 and 1")
    parser.add_argument("--shadow", required=False, action="store_true", help="Whether or not shadows should be added to the Sign")
    parser.add_argument("--shadow_frequency", required=False, default=2, type=float, help="How many different shadow/light patches are added, should be between 1 and 5")
    parser.add_argument("--shadow_intensity", required=False, default=0.3, type=float, help="Defines the intensity of the shadow, should be between 0 and 1")
    parser.add_argument("-z", "--zoom", required=False, action="store_true", help="Whether or not the program zooms on the sign")
    parser.add_argument("-zp", "--zoom_position", required=False, nargs=2, default=[0.0, 0.0], type=float, help="Position parameters on where to zoom on")
    parser.add_argument("-zf", "--zoom_factor", required=False, nargs=2, default=[1, 1], type=float, help="Upper and Lower limits of the zoom factor")
    parser.add_argument("-m", "--monochrome", required=False, action="store_true", help="Whether or not to apply a random monochrom light")
    parser.add_argument("-sf", "--shear_factor", required=False, nargs=2, default=[0, 0], type=float, help="Determines the min and max factor the Sign is sheared, 0 is nothing 1 is 45°")
    parser.add_argument("--export_config", required=False, action="store_true", help="Set this flag to export the config.yaml into the output folder")

    args = parser.parse_args()

    if args.config is not None:
        try:
            os.path.isfile(args.config)
        except OSError:
            parser.error("Path to config file is not valid")
        with open(args.config, 'r') as file:
            config = yaml.safe_load(file)
        final_config = {}
        # Use values from config file as base
        final_config.update(config)
        # Override with command line arguments if present
        #final_config.update({k: v for k, v in vars(args).items() if v is not None})
        final_config["export_config"] = args.export_config
        args = argparse.Namespace(**final_config)

    check_required_args(parser, args)
    validate_args(parser, args)
    validate_paths(parser, args)

    return args

def check_required_args(parser, args):
    required = ["signs", "cars", "output", "format", "number"]
    for arg in required:
        if arg not in args or vars(args)[arg] is None:
            parser.error(f"Value {arg} is required!")

def validate_args(parser, args):
    if args.rotation:
        if args.rotation[0] < -180 or args.rotation[1] > 180:
            parser.error("Rotation must be between -180 and 180")
        if args.rotation[0] > args.rotation[1]:
            parser.error("The first value (minimum) can not be larger than the second value (maximum)")
    if args.size:
        if args.size[0] <= 0:
            parser.error("Size must be greater than 0")
        if args.size[0] > args.size[1]:
            parser.error("The first value (minimum) can not be larger than the second value (maximum)")

    if args.position:
        if args.position < 0 or args.position > 500:
            parser.error("Position variation should be between 0 and 500")

def validate_paths(parser, args):
    try:
        os.path.isdir(args.signs)
    except OSError:
        parser.error("Path to sign images is not valid")
    
    try:
        os.path.isdir(args.cars)
    except OSError:
        parser.error("Path to car images is not valid")
    
    try:
        os.path.isdir(args.output)
    except OSError:
        parser.error("Output path is not valid")

def get_random_image(directory_path, image_format, random_gen: random.Random):
    """
    Selects a random image from the specified directory.

    Parameters:
        directory_path (str): The path to the directory containing the images.
        image_format (str): The format of the images (e.g., '.jpg', '.png').

    Returns:
        str: The path to a randomly selected image.
    """
    # Get a list of all files in the directory that have the specified format
    images = glob.glob(f"{directory_path}/*.{image_format}")
    
    if not images:
        raise Exception(f"No images of format {image_format} found in directory {directory_path}")
    
    # Select a random image from the list
    image = random_gen.choice(images)
    # Return the full path to the image
    return os.path.abspath(image)

def get_random_from_bounds(lower, upper, random_gen: random.Random):
    if type(lower) == type(upper):
        if isinstance(lower, int):
            return random_gen.randint(lower, upper)
        elif isinstance(lower, float):
            return random_gen.uniform(lower, upper)
        else:
            raise TypeError("type_value must be an instance of int or float.")
    else:
        raise TypeError("Lower and Upper value cant be a different type.")
    

def handle_xml(filename):
    xml_file = filename.split(".")[-2] + ".xml"
    tree = etree.parse(xml_file)
    root = tree.getroot()

    # Extract the points from the XML
    points = root.xpath('//point')
    point_list = [[(int(float(point.xpath('x/text()')[0])), int(float(point.xpath('y/text()')[0])))] for point in points]

    # Convert the points to a NumPy array
    points = np.array(point_list)

    return points

def export_config(args, path, filename):
    args_dict = vars(args)
    with open(os.path.join(path, filename), "w+") as file:
        del args_dict["export_config"]
        del args_dict["config"]
        yaml.dump(args_dict, file, default_flow_style=False)