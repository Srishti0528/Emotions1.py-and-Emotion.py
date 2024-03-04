# here in this code we are using a dataset and then if any face matches the face described in the dataset then the same is displayed along with the emotions. 

import cv2
import os
import torch
import numpy as np
from deepface import DeepFace
from facenet_pytorch import InceptionResnetV1, MTCNN
import torch.nn.functional as F

# Load the pre-trained emotion detection model
emotion_model = DeepFace.build_model("Emotion")

# Define emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'] 
 
# Load pre-trained facenet model for face recognition
facenet = InceptionResnetV1(pretrained='vggface2').eval()

# Load MTCNN model for face detection
mtcnn = MTCNN()

# Create a database of known faces
database = {}
for name in os.listdir('C:\\Users\\HP\\Downloads\\Facial-Emotion-Recognition-with-OpenCV-and-Deepface-main\\Facial-Emotion-Recognition-with-OpenCV-and-Deepface-main\\archive\\New folder (3)\\New folder (3)'):   
    img_path = cv2.imread(f'C:\\Users\\HP\\Downloads\\Facial-Emotion-Recognition-with-OpenCV-and-Deepface-main\\Facial-Emotion-Recognition-with-OpenCV-and-Deepface-main\\archive\\New folder (3)\\New folder (3)\\{name}') 
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.imread(img_path) 
    if img is None:
        print(f"Could not read image at {img_path}")
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (160, 160))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    # img = torch.Tensor(img)
    img = torch.Tensor(img)
    img = img.permute(0, 3, 1, 2)  # rearrange dimensions to (batch_size, channels, height, width)
    embedding = facenet(img).detach().numpy()
    # embedding = facenet(img).detach().numpy()
    database[name] = embedding

# Start capturing video
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    

    if ret is False:
        print("Failed to read frame")
        continue
    # Convert frame to grayscale for emotion detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Use MTCNN to detect faces in the frame
    faces = mtcnn.detect(frame)
 
    if faces[0] is not None:
        for box in faces[0]:
            x, y, w, h = box

            # Extract the face ROI from the colored frame for FaceNet
            face_roi_color = frame[int(y):int(h), int(x):int(w)]

            # Extract the face ROI from the grayscale frame for Emotion Detection
            face_roi_gray = gray_frame[int(y):int(h), int(x):int(w)]

            # Resize the face ROI to match the input shape of the emotion model
            # resized_face = cv2.resize(face_roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            if face_roi_gray.size == 0:
                print(f"No face detected in image {name}")
            else:
                resized_face = cv2.resize(face_roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
                # rest of your code

            # Normalize the resized grayscale face image
            normalized_face = resized_face / 255.0

            # Reshape the grayscale image to match the input shape of the emotion model
            reshaped_face = normalized_face.reshape(1, 48, 48, 1)

            # Predict emotions using the pre-trained model
            emotion_preds = emotion_model.predict(reshaped_face)[0]
            emotion_idx = emotion_preds.argmax()
            emotion = emotion_labels[emotion_idx]

            # Draw rectangle around face and label with predicted emotion
            cv2.rectangle(frame, (int(x), int(y)), (int(w), int(h)), (0, 0, 255), 2)
            cv2.putText(frame, emotion, (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            

        # Generate a face embedding for the color face ROI
            face_roi_color = cv2.resize(face_roi_color, (160, 160))
            face_roi_color = face_roi_color / 255.0
            face_roi_color = np.expand_dims(face_roi_color, axis=0)
            face_roi_color = torch.Tensor(face_roi_color)
            face_roi_color = face_roi_color.permute(0, 3, 1, 2)  # rearrange dimensions to (batch_size, channels, height, width)
            embedding = facenet(face_roi_color).detach().numpy()

            # Compare the face embedding with the database of known faces
            min_distance = float('inf')
            recognized_name = None
            for name, known_embedding in database.items():
                distance = F.pairwise_distance(torch.Tensor(embedding), torch.Tensor(known_embedding))
                if distance < min_distance:
                    min_distance = distance
                    recognized_name = name

            # If the face was recognized, add the name to the frame
            if recognized_name is not None:
                cv2.putText(frame, recognized_name, (int(x), int(y) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Real-time Emotion Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()