from picamera2 import Picamera2
import time
from PIL import Image, ImageDraw, ImageFont
import tflite_runtime.interpreter as tflite
import numpy as np
from label_map_coco import LABEL_MAP

# Define a dictionary to map each class to a specific color
CLASS_COLORS = {
    1: "red",    # Person
    2: "blue",   # Bicycle
    3: "green",  # Car
    4: "purple", # Motorbike
    5: "orange", # Airplane
    # Add more mappings as needed
    # Default to yellow for other classes
}

def capture_image():
    """Capture an image using the Picamera2 and save it as 'test_image.jpg'."""
    # Create a Picamera2 instance
    picam2 = Picamera2()

    # Configure the camera settings
    preview_config = picam2.create_preview_configuration(main={"format": 
                                                               'XRGB8888', "size": (300, 300)})
    picam2.configure(preview_config)

    # Start the camera
    picam2.start()
    print("Camera started. Warming up...")
    time.sleep(0.5)  # Give the camera some time to warm up

    # Capture an image to a numpy array
    image = picam2.capture_array()
    print("Image captured.")

    # Convert the image to a PIL Image object
    img = Image.fromarray(image)

    # If the image has an alpha channel, convert it to RGB
    if img.mode == 'RGBA':
        img = img.convert('RGB')

    # Save the image to a file
    img.save('test_image.jpg')
    print("Image saved as 'test_image.jpg'.")

    # Stop the camera
    picam2.stop()
    print("Camera stopped.")

def load_interpreter(model_path):
    """Load the TFLite model interpreter."""
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def draw_bounding_boxes(image, boxes, classes, scores, confidence_threshold):
    """Draw bounding boxes and labels on the image."""
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()  # Use the default font

    width, height = image.size
    for i in range(len(scores[0])):
        if scores[0][i] > confidence_threshold:
            ymin, xmin, ymax, xmax = boxes[0][i]
            (left, right, top, bottom) = (xmin * width, xmax * width, ymin * height, ymax * height)
            class_id = int(classes[0][i])
            color = CLASS_COLORS.get(class_id, "yellow")  # Default to yellow if class not found
            label = f"{LABEL_MAP.get(class_id, 'unlabeled')}: {scores[0][i]:.2f}"
            draw.rectangle([(left, top), (right, bottom)], outline=color, width=2)
            draw.text((left, top), label, fill=color, font=font)

    return image

def run_object_detection(image_path, model_path, confidence_threshold=0.5):
    """Run object detection on the given image using the specified model."""
    # Load the image
    image = Image.open(image_path)
    
    # Load TFLite model and allocate tensors
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']

    # Debug: Print output tensor details
    for output in output_details:
        print(output['name'], "shape:", output['shape'], "dtype:", output['dtype'])

    # Prepare input data
    input_image = image.resize((input_shape[1], input_shape[2]))
    input_image = np.expand_dims(input_image, axis=0)
    if input_details[0]['dtype'] == np.float32:
        input_image = (np.float32(input_image) - 127.5) / 127.5

    # Set the model input
    interpreter.set_tensor(input_details[0]['index'], input_image)

    # Run inference
    interpreter.invoke()

    # Retrieve outputs
    boxes = interpreter.get_tensor(output_details[0]['index'])
    classes = interpreter.get_tensor(output_details[1]['index'])  # Class labels
    scores = interpreter.get_tensor(output_details[2]['index'])   # Confidence scores

    # Draw bounding boxes and labels on the image
    image_with_boxes = draw_bounding_boxes(image, boxes, classes, scores, confidence_threshold)
    
    # Save the image with bounding boxes
    image_with_boxes.save('output_image.jpg')
    print("Output image saved as 'output_image.jpg'.")

    # Process outputs
    print("Detected objects:")
    for i in range(len(scores[0])):
        if scores[0][i] > confidence_threshold:
            ymin, xmin, ymax, xmax = boxes[0][i]
            print(f"Class {LABEL_MAP.get(int(classes[0][i]), 'unlabeled')}, Confidence: {scores[0][i]:.2f}, Bounding box: ({ymin:.2f}, {xmin:.2f}, {ymax:.2f}, {xmax:.2f})")

if __name__ == "__main__":
    model_path = 'efficientdet_lite0.tflite'

    # Load the model interpreter
    interpreter = load_interpreter(model_path)

    # Capture an image
    capture_image()  # This saves 'test_image.jpg'

    # Run object detection on the captured image
    run_object_detection('test_image.jpg', 
                         model_path=model_path, confidence_threshold=0.3)
