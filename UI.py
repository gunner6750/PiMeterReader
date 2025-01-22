from flask import Flask, render_template, send_from_directory, request,render_template, jsonify, request
import os
from datetime import datetime
from collections import defaultdict
app = Flask(__name__)

# Folder where processed images and labels are stored
IMAGE_FOLDER = 'output'

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/get_image_counts_for_month', methods=['GET'])
def get_image_counts_for_month():
    # Fetch all images and group them by date
    image_counts = defaultdict(int)
    for filename in os.listdir(IMAGE_FOLDER):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(IMAGE_FOLDER, filename)
            modified_time = datetime.fromtimestamp(os.path.getmtime(image_path)).date()

            # Group by the modified date
            image_counts[modified_time] += 1

    # Prepare the events in a format that FullCalendar can use
    events = [{'date': date.strftime('%Y-%m-%d'), 'count': count} for date, count in image_counts.items()]
    return jsonify({'events': events})

@app.route('/images/<filename>')
def serve_image(filename):
    return send_from_directory(IMAGE_FOLDER, filename)

@app.route('/date/<selected_date>', methods=['GET', 'POST'])
def view_images_by_date(selected_date):
    show_with_label = True
    if request.method == 'POST':
        # Check if the 'with_label' checkbox is checked
        show_with_label = 'with_label' in request.form

    # Get the list of images for the selected date
    files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # List to store image paths and labels for the selected date
    image_data = []

    for file in files:
        image_path = os.path.join(IMAGE_FOLDER, file)
        label_file = file.rsplit('.', 1)[0] + '.txt'
        label_path = os.path.join(IMAGE_FOLDER, label_file)

        label = ''
        if os.path.exists(label_path):
            try:
                with open(label_path, 'r', encoding='utf-8') as f:
                    label = f.read().strip()
            except Exception as e:
                print(f"Unexpected error with file {label_path}: {e}")
                label = "Error reading file"

        try:
            modified_time = datetime.fromtimestamp(os.path.getmtime(image_path))
        except FileNotFoundError:
            print(f"File not found: {image_path}")
            continue

        date_str = modified_time.date().strftime('%Y-%m-%d')

        if date_str == selected_date:
            # Filter images based on the 'show_with_label' option
            if show_with_label and not label:
                continue  # Skip images without labels if the checkbox is checked
            image_data.append({
                'image': file,
                'label': label,
                'modified_time': modified_time
            })

    # Sort images by modified time (latest first)
    image_data.sort(key=lambda x: x['modified_time'], reverse=True)

    return render_template('view_date.html', selected_date=selected_date, images=image_data, show_with_label=show_with_label)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
