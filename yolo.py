import cv2
import numpy as np
import json
import os
import argparse

# Parse command-line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections, IoU threshold")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())

# Load YOLO
net = cv2.dnn.readNetFromDarknet('yolo-coco/yolov3.cfg', 'yolo-coco/yolov3.weights')
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load class names
with open('yolo-coco/coco.names', 'r') as f:
    classes = [line.strip() for line in f]

# Function to perform object detection
def detect_objects(image):
    height, width, channels = image.shape

    # Create blob from image
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    
    net.setInput(blob)

    # Forward pass through the network
    outs = net.forward(output_layers)

    # Information to be extracted from the detection
    class_ids = []
    confidences = []
    boxes = []

    # Extract information from the output
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > args["confidence"]:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, x+w, y+h])  # Format [x1, y1, x2, y2]
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
    if len(boxes) > 0:
        # Ensure that both boxes and confidences are non-empty and have the correct format
        if all(isinstance(box, (list, tuple)) and len(box) == 4 for box in boxes) and all(isinstance(conf, float) for conf in confidences):
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])
            return [class_ids[i] for i in idxs.flatten()], [confidences[i] for i in idxs.flatten()], [boxes[i] for i in idxs.flatten()]
        else:
            print("Error: Invalid format for boxes or confidences.")
            return [], [], []
    else:
        return [], [], []

# Main function
def main():
    images_folder = 'images'
    output_json_path = 'output_test.json'

    results = []

    for frame_number, image_name in enumerate(os.listdir(images_folder), start=1):
        image_path = os.path.join(images_folder, image_name)
        image = cv2.imread(image_path)

        class_ids, confidences, boxes = detect_objects(image)

        # Print debug information
        print(f"Processing image: {image_name}")
        print("Detected class IDs:", class_ids)
        print("Confidences:", confidences)
        print("Detected boxes:", boxes)

        # Store the detection results in the desired format
        frame_result = {
            "frame_number": frame_number,
            "objects": boxes,
            "no_of_objects": len(boxes),
            "image_name": image_name
        }

        results.append(frame_result)

    print("Results:", results)

    # Save the results to a JSON file
    with open(output_json_path, 'w') as json_file:
        json.dump(results, json_file, indent=2, default=str)

if __name__ == "__main__":
    main()
