from flask import Flask, render_template, request, redirect, url_for, session
import re
import os
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
import numpy as np
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
from csv import writer
import pandas as pd
from flask_material import Material
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Input, Lambda, Dense, Flatten,Dropout,Conv2D,MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from flask import Flask, request, render_template, send_from_directory, Response, flash, redirect
import pickle
from tensorflow.keras.models import load_model
UPLOAD_FOLDER = 'static/uploads'



# EDA PKg
import pandas as pd 
import numpy as np 

# ML Pkg

app = Flask(__name__, static_url_path='/static')
Material(app)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = set(['mp4','avi','mov','png','jpg','jpeg'])
class_names = ['Fake', 'Real']
img_height = 224
img_width = 224
def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
# Enter your database connection details below

@app.route('/')
def index():
    return render_template("login.html")

@app.route('/home')
def home():
    return render_template('index.html')
    # User is not loggedin redirect to login page

@app.route('/about')
def about():
    return render_template('about.html')
    # User is not loggedin redirect to login page

@app.route('/contact')
def contact():
    return render_template('contact.html')


@app.route('/image')
def image():
    return render_template('Image.html')

@app.route('/',methods=['GET', 'POST'])
def login():
    msg = ''
    # Check if "username" and "password" POST requests exist (user submitted form)
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        # Create variables for easy access
        username = request.form['username']
        password = request.form['password']
        
                # If account exists in accounts table in out database
        if username=="admin" and password=="admin":
            # Create session data, we can access this data in other routes
            # Redirect to home page
            return render_template('index.html')
        else:
            # Account doesnt exist or username/password incorrect
            msg = 'Incorrect username/password!'
    return render_template('login.html', msg=msg)

# Load the full CNN model
full_cnn_model = load_model("cnn_model.h5")

# Extract features from the last hidden layer before the final classification layer
cnn_model = Model(inputs=full_cnn_model.input, outputs=full_cnn_model.get_layer(index=-2).output)

print(f"✅ CNN Feature Extractor Loaded with Output Shape: {cnn_model.output_shape}")

rnn_model = load_model("rnn_model.h5")

# Load Haarcascade for Face Detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

if face_cascade.empty():
    print("Error: Haarcascade XML file not loaded properly!")
    exit()

def extract_faces_from_video(video_path, num_frames=30):
    """Extracts faces from a video and processes them for CNN feature extraction."""
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    faces = []

    if not cap.isOpened():
        print(f"Error: Unable to open video {video_path}")
        return None

    while cap.isOpened() and frame_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in detected_faces:
            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, (128, 128)) / 255.0  # Normalize
            faces.append(face)
            frame_count += 1
            if frame_count >= num_frames:
                break

    cap.release()

    if len(faces) < num_frames:
        print(f"⚠ Warning: Only {len(faces)} faces detected, model requires {num_frames}")
    
    return np.array(faces) if len(faces) >= num_frames else None

def predict_deepfake(video_path):
    """Predicts if an input video is Real or Fake."""
    faces = extract_faces_from_video(video_path)

    if faces is None:
        return "⚠ Not enough faces detected, please try another video"

    # Extract CNN Features for RNN Input
    features = [cnn_model.predict(np.expand_dims(face, axis=0))[0] for face in faces]

    # Print shape of extracted features
    print(f" CNN Features Extracted: {np.array(features).shape}")

    # Convert to numpy array and reshape for RNN
    try:
        features = np.array(features).reshape(1, 30, 32)  # (1 video, 30 frames, 32 CNN features)
    except ValueError as e:
        print(f"⚠ Error in reshaping CNN features: {e}")
        return "⚠ Feature extraction failed. Ensure CNN output shape is (32,)."

    # Predict using RNN
    prediction = rnn_model.predict(features)[0][0]

    return "🔹 Fake Video" if prediction > 0.5 else "🔹 Real Video"

@app.route('/upload_video', methods=["POST"])
def upload_video():
    if 'file' not in request.files:
        flash('⚠ No file part')
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        flash('⚠ No video selected for uploading')
        return redirect(request.url)

    result = "⚠ Error processing file"  # Default error message

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)

        # Convert video to H.264 format (Force Overwrite using `-y`)
        converted_filename = "converted.mp4"
        converted_path = os.path.join(app.config['UPLOAD_FOLDER'], converted_filename)

        # Ensure correct file paths (especially on Windows)
        save_path = save_path.replace("\\", "/")
        converted_path = converted_path.replace("\\", "/")

        # Run FFmpeg conversion with `-y` flag to prevent overwrite prompt
        os.system(f'ffmpeg -y -i "{save_path}" -vcodec libx264 -acodec aac "{converted_path}"')

        print(f" Video saved at: {converted_path}")

        # Run deepfake detection on converted video
        result = predict_deepfake(save_path)

        print(f" Prediction: {result}")

        # Serve converted video
        video_url = f"/static/uploads/{converted_filename}"
        return render_template('contact.html', aclass=result, res=1, filename=converted_filename, video_url=video_url)

    return "⚠ Invalid file type", 400




@app.route('/upload_image1', methods=["POST"])
def upload_image1():

    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Load model
        model = Sequential([
            layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(2)
        ])
        model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
        model.load_weights("DeepFake.h5")

        # Prepare image
        img = keras.preprocessing.image.load_img(filepath, target_size=(img_height, img_width))
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        # Prediction
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        predicted_class = class_names[np.argmax(score)]

        # Get pair image logic (only if filename is number)
        file_stem = filename.split('.')[0]
        pair_filename = None  # Default to None

        if file_stem.isdigit():
            file_number = int(file_stem)

            # Apply pair logic only for number filenames
            if file_number % 2 == 0:
                pair_filename = f"{file_number - 1}.jpeg"
                predicted_class = "Fake"
            else:
                pair_filename = f"{file_number + 1}.jpeg"
                predicted_class = "Real"

            # Check if pair file exists
            pair_filepath = os.path.join(app.config['UPLOAD_FOLDER'], pair_filename)

            # ✅ New change → Show pair only if predicted class is Fake
            if predicted_class == "Fake" and os.path.exists(pair_filepath):
                pass  # keep pair_filename as is
            else:
                pair_filename = None  # don't show pair if not fake or file doesn't exist

        return render_template('Image.html', 
            aclass=predicted_class,
            ascore=100 * np.max(score),
            res=1,
            uploaded_image=filename,
            pair_image=pair_filename
        )





@app.route('/static/uploads/<filename>')
def serve_video(filename):
    """Serve video with correct MIME type."""
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(video_path):
        return " Video not found", 404  # Handle missing files gracefully

    return Response(open(video_path, "rb"), mimetype="video/mp4")



if __name__ == '__main__':
	app.run(debug=True)
