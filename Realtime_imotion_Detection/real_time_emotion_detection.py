import cv2
import numpy as np
from tensorflow.keras.models import load_model

def real_time_emotion_detection(model_path):
    model = load_model(model_path, compile=False)
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Emotion Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Emotion Detection", 800, 600)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))
            face = face.astype('float32') / 255.0
            face = np.expand_dims(face, axis=-1)
            face = np.expand_dims(face, axis=0)
            
            emotion_prediction = model.predict(face)
            max_index = np.argmax(emotion_prediction[0])
            emotion_label = emotion_labels[max_index]
            
            cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        cv2.imshow('Emotion Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

real_time_emotion_detection('emotion_detection_model.h5')
