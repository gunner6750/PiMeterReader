import tensorflow as tf
import numpy as np
import cv2
import os
import time
# Path to your TFLite model
model_path = 'tflite_model'

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
print("Model loaded successfully!")
# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input details:", input_details)
print("Output details:", output_details)
def non_max_suppression(predictions, iou_threshold=0.5):
    """
    Perform Non-Maximum Suppression (NMS) on predictions.
    Args:
        predictions (list of tuples): Each tuple is (class_id, score, x_min, y_min, x_max, y_max).
        iou_threshold (float): IoU threshold for suppression.
    Returns:
        list of tuples: Filtered predictions after NMS.
    """
    # Sort predictions by confidence score in descending order
    predictions = sorted(predictions, key=lambda x: x[1], reverse=True)
    filtered_predictions = []

    while predictions:
        # Pick the box with the highest confidence
        chosen_box = predictions.pop(0)
        filtered_predictions.append(chosen_box)

        # Compare the chosen box with the rest
        predictions = [
            box
            for box in predictions
            if calculate_iou(chosen_box, box) < iou_threshold
        ]

    return filtered_predictions

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two boxes.
    Args:
        box1, box2 (tuple): Each tuple is (class_id, score, x_min, y_min, x_max, y_max).
    Returns:
        float: IoU value.
    """
    # Extract coordinates
    _, _, x1_min, y1_min, x1_max, y1_max = box1
    _, _, x2_min, y2_min, x2_max, y2_max = box2

    # Calculate the intersection
    x_inter_min = max(x1_min, x2_min)
    y_inter_min = max(y1_min, y2_min)
    x_inter_max = min(x1_max, x2_max)
    y_inter_max = min(y1_max, y2_max)

    if x_inter_min >= x_inter_max or y_inter_min >= y_inter_max:
        return 0.0

    intersection_area = (x_inter_max - x_inter_min) * (y_inter_max - y_inter_min)

    # Calculate the union
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - intersection_area

    return intersection_area / union_area
def process_predictions_with_nms(output_data, image, conf_threshold=0.25, iou_threshold=0.2):
    h, w, _ = image.shape
    predictions = []

    # Parse predictions
    for prediction in output_data:
        x_center, y_center, width, height = prediction[:4]
        obj_score = prediction[4]
        class_probs = prediction[5:]

        # Filter by confidence threshold
        if obj_score < conf_threshold:
            continue

        # Convert class probabilities to class ID
        class_id = np.argmax(class_probs)
        class_score = class_probs[class_id]

        # Convert normalized bbox to pixel coordinates
        x_min = int((x_center - width / 2) * w)
        y_min = int((y_center - height / 2) * h)
        x_max = int((x_center + width / 2) * w)
        y_max = int((y_center + height / 2) * h)

        predictions.append((class_id, class_score, x_min, y_min, x_max, y_max))
    print('boxes recognized:',len(predictions))
    # Apply Non-Maximum Suppression
    filtered_predictions = non_max_suppression(predictions, iou_threshold)

    # Draw the filtered predictions
    for class_id, class_score, x_min, y_min, x_max, y_max in filtered_predictions:
        label = f"Class {class_id}: {class_score:.2f}"
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image, filtered_predictions

def calculate_centroid_from_detection(detection):
    """
    Calculates the centroid of a bounding box from detection input.
    
    Args:
        detection (tuple): A tuple (index, confidence, x_min, y_min, x_max, y_max).
    
    Returns:
        tuple: (x_center, y_center) of the box.
    """
    _, _, x_min, y_min, x_max, y_max = detection
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    return x_center, y_center

def cluster_detections(detections, distance_threshold=200):
    """
    Clusters detections based on the distance between their centroids.
    
    Args:
        detections (list of tuples): List of detections with format 
                                     (index, confidence, x_min, y_min, x_max, y_max).
        distance_threshold (float): Maximum distance for two boxes to be in the same cluster.
    
    Returns:
        list of lists: A list where each sublist contains the indices of detections in the same cluster.
    """
    # Calculate centroids of all detections
    centroids = np.array([calculate_centroid_from_detection(detection) for detection in detections])
    n = len(centroids)
    
    # Initialize clustering
    clusters = []
    visited = set()
    
    def find_neighbors(index):
        """Finds all neighbors of a given detection index within the distance threshold."""
        neighbors = []
        for j in range(n):
            if j != index and j not in visited:
                dist = np.linalg.norm(centroids[index] - centroids[j])
                if dist <= distance_threshold:
                    neighbors.append(j)
        return neighbors

    # Perform clustering
    for i in range(n):
        if i not in visited:
            # Start a new cluster
            cluster = [i]
            visited.add(i)
            # Queue for BFS-style exploration
            queue = [i]
            while queue:
                current = queue.pop(0)
                neighbors = find_neighbors(current)
                for neighbor in neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        cluster.append(neighbor)
                        queue.append(neighbor)
            clusters.append(cluster)
    
    return clusters


def extract_numbers_from_clusters(detections, clusters, min_members=3):
    """
    Extracts numbers from clusters with more than the specified number of members.

    Args:
        detections (list of tuples): List of detections with format 
                                     (class_id, confidence, x_min, y_min, x_max, y_max).
        clusters (list of lists): Clustered indices of detections.
        min_members (int): Minimum number of members required to process a cluster.

    Returns:
        list: Concatenated numbers (as strings) from the processed clusters.
    """
    results = []
    for cluster in clusters:
        if len(cluster) >= min_members:
            # Sort the cluster indices by x_min (detections[idx][2])
            sorted_predictions = sorted(cluster, key=lambda idx: detections[idx][2])
            # Extract digits (class_id) from sorted detections
            digits = [str(int(detections[idx][0])) for idx in sorted_predictions]
            # Concatenate digits to form the number
            concatenated_number = "".join(digits)
            results.append(concatenated_number)
    return results
def process_image(image_path):
    # Load the image
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (640, 640))  # Resize to the model's input size
    img_normalized = img_resized / 255.0      # Normalize pixel values to [0, 1]
    img_expanded = np.expand_dims(img_normalized, axis=0).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], img_expanded)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]

    processed_image, predictions = process_predictions_with_nms(output_data, img)
    clusters = cluster_detections(predictions)
    result = extract_numbers_from_clusters(predictions, clusters, min_members=3)
    print("number recognized: ",result)
    txt = os.path.basename(image_path).replace("jpg", "txt")

    f = open("output/"+txt,"w")
    f.write("\n".join(result))
    f.close()
    # Return the processed image
    return processed_image

def process_new_files(input_dir, output_dir):
    processed_files = set()  # To track processed files

    # Initial scan to populate the processed_files set with existing files
    existing_files = os.listdir(input_dir)
    for file_name in existing_files:
        file_path = os.path.join(input_dir, file_name)
        if os.path.isfile(file_path) and file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            processed_files.add(file_name)

    print("Starting to check for new images...")

    while True:
        # List all files in the input directory
        files = os.listdir(input_dir)

        # Process files that haven't been processed yet
        for file_name in files:
            file_path = os.path.join(input_dir, file_name)

            # Check if the file is an image and hasn't been processed yet
            if os.path.isfile(file_path) and file_name.lower().endswith(('.png', '.jpg', '.jpeg')) and file_name not in processed_files:
                print(f"New image detected: {file_path}")
                try:
                    # Process the image
                    processed_image = process_image(file_path)

                    # Save the processed image
                    output_path = os.path.join(output_dir, file_name)
                    cv2.imwrite(output_path, processed_image)
                    print(f"Processed image saved to: {output_path}")

                    # Add the file to processed set
                    processed_files.add(file_name)
                except Exception as e:
                    print(f"Error processing image: {e}")

        # Wait for 5 seconds before checking for new files
        time.sleep(5)

if __name__ == "__main__":
    input_dir1 = "camera/Internal shared storage/DCIM/Camera"  # Folder where images from your phone appear
    input_dir2 = "camera/Bộ nhớ trong dùng chung/DCIM/Camera"

    output_dir = "output"  # Folder to save processed images

    # Ensure the output folder exists
    os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(input_dir1):
        input_dir = input_dir1
    else:
        input_dir = input_dir2
    print("Checking for new images every 5 seconds...")
    process_new_files(input_dir, output_dir)
