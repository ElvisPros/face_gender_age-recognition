import cv2
import dlib
from PIL import Image
from keras.models import load_model
import numpy as np

# Load pre-trained face detector
face_detector = dlib.get_frontal_face_detector()

# Load pre-trained age and gender detection models
age_model = load_model('path_to_age_model.h5')  # Provide the path to your age model
gender_model = load_model('path_to_gender_model.h5')  # Provide the path to your gender model

# Function to detect age and gender
def detect_age_gender(face):
    # Preprocess face image
    face = cv2.resize(face, (64, 64))
    face = np.expand_dims(face, axis=0)
    
    # Predict age
    predicted_age = age_model.predict(face)[0][0]
    
    # Predict gender
    predicted_gender = gender_model.predict(face)[0][0]
    gender = "Male" if predicted_gender < 0.5 else "Female"
    
    return predicted_age, gender

# Function to detect faces in an image
def detect_faces(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    faces = face_detector(gray)
    
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Extract face ROI
        face_roi = image[y:y+h, x:x+w]
        
        # Detect age and gender
        age, gender = detect_age_gender(face_roi)
        
        # Display age and gender
        cv2.putText(image, f"Age: {int(age)}", (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(image, f"Gender: {gender}", (x, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Display the result
    image = Image.fromarray(image)
    image.show()

# Example usage
detect_faces 
r'C:\Users\ELVIS PROS\Desktop\computer vision projects\face,age and gender detection\elvis.jpeg'  # Provide the path to your test image
