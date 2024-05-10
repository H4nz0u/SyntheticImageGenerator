import random
import cv2
import os
from tqdm import tqdm
from utils import parse_arguments, get_random_image, handle_xml, export_config
from PascalVOCAnnotator import PascalVOCAnnotation
from SignImage import SignImage
from CarImage import CarImage
import concurrent.futures
import traceback


def write(carImage: CarImage, signImage: SignImage, args):
    hash_value = hex(abs(hash(carImage.image.tobytes())))[8:]

    carImage.write(args.output, f"{hash_value}.{args.format}")

    annotator = PascalVOCAnnotation(
        f"{hash_value}.{args.format}", args.output, carImage.image.shape
    )

    annotator.append_object(signImage.bounding_box, "Typlabel-China")
    annotator.write_xml(os.path.join(args.output, f"{hash_value}.xml"))


def process_image(i, args):
    random_gen = random.Random(args.seed + i)
    # Get random sign and car images
    sign_image_file = get_random_image(args.signs, args.format, random_gen)
    car_image_file = get_random_image(args.cars, args.format, random_gen)

    # Read sign and car images
    sign_image = cv2.imread(sign_image_file)
    car_image = cv2.imread(car_image_file)

    # Extract bounding box from XML file
    bounding_box = handle_xml(sign_image_file)

    signImage = SignImage(sign_image, bounding_box, random_gen, **vars(args))

    signImage.process()

    # Get random rotation angle and position for placing the sign

    carImage = CarImage(car_image, signImage, random_gen, **vars(args))

    # Place sign on car and add noise
    carImage.process()

    write(carImage, signImage, args)


def main():
    """
    Main function.
    """
    # Parse arguments
    args = parse_arguments()
    if args.export_config and args.config is None:
        export_config(args, args.output, "config.yaml")
    pbar = tqdm(total=args.number, desc="Processing images", unit="image")

    random.seed(args.seed)
    print(f"seed: {args.seed}")
    # Initialize progress bar
    # Process images
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_image, i, args) for i in range(args.number)]
        for future in futures:
            # Add a "done" callback to each future. This will be called when the future completes.
            try:
                # Try to get the result of the future. If the function raised an exception,
                # this will re-raise that exception here.
                future.result()
            except Exception as e:
                # The function raised an exception, so we print the exception and cancel all other futures
                print(f"An exception occurred:")
                traceback.print_exc()
                for future in futures:
                    future.cancel()
                break
            else:
                # No exception occurred, so update the progress bar
                pbar.update(1)


if __name__ == "__main__":
    main()
