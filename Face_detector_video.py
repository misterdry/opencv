import cv2

from random import randrange

trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #looks like if the image is to small the code does not recognice the face

# To capture face from webcam
# webcam = cv2.VideoCapture('Bros Before Hos Explained by Michael Scott.mp4')
webcam = cv2.VideoCapture(0)

#Iterate forever over frames, it is doing all the work in real time
while True:
    
    #Read the current frame
    succesful_frame_read, frame = webcam.read()
    
    # Must convert it to grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #it is need to write BGR not RGB
    
    # Detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
   
    for (x, y, w, h) in face_coordinates:
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    cv2.imshow('Misterdry Face Detector', frame)
    # key = cv2.waitKey(100) #on videos if no number is introduce "makes photos"
    cv2.waitKey(1)
    
    # # Stop if Q key is press
    # if key==81 or key==113:
    #     break
