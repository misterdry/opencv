import cv2
import numpy as np

# Face and smile classsifier
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_detector = cv2.CascadeClassifier('haarcascade_smile.xml')
eye_detector = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')

# Archives
webcam = cv2.VideoCapture(0)

# Show the current frame
while True:
    
    # Read the current frame from the webcam video
    succesfull_frame_read, frame = webcam.read()
    
    # If there  is an error, abort
    if not succesfull_frame_read:
        break
    
    # Change to grayscale
    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces first
    faces = face_detector.detectMultiScale(frame_grayscale)
    # smiles = smile_detector.detectMultiScale(frame_grayscale, scaleFactor = 1.7, 
                                             # minNeighbors = 20) # Blur the image to make
    #easy to see the objects and not confuse, neighbors se refiere al 
    #numero de rectángulos mínimo para que sea una sonrisa
    
    # Draw rectangles
    for (x, y, w, h) in faces:
        
        cv2.rectangle(frame, (x, y), (x + w, y +h), (100, 200, 50), 4)
        
        # The_face localization / subframe
        the_face = frame[y:y+h, x:x+w] # Opencv numpy arrays by default
        
        # Change to grayscale
        face_grayscale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)
        
        smiles = smile_detector.detectMultiScale(face_grayscale, scaleFactor = 1.7, minNeighbors = 20) # Blur the image to make easy to see the objects and not confuse, neighbors se refiere al 
        #numero de rectángulos mínimo para que sea una sonrisa
        
        eyes = eye_detector.detectMultiScale(face_grayscale, 1.3, 10)
        
        # Label this face as smiling
        if len(smiles) > 0:
            cv2.putText(frame, 'smliling', (x, y + h + 40), fontScale = 3, 
                        fontFace = cv2.FONT_HERSHEY_PLAIN, color = (0, 0, 0))
        
        # Label this face as eying
        if len(eyes) > 0:
            cv2.putText(frame, 'Caucasian', (x, y + h + 90), fontScale = 3, 
                        fontFace = cv2.FONT_HERSHEY_PLAIN, color = (0, 0, 0))
        
        # Find all smiles in the face
        for (x_, y_, w_, h_)  in smiles:
                
                # Draw a rectangle around a smile
                cv2.rectangle(the_face, (x_, y_), (x_ + w_, y_ + h_), (50, 50, 200), 4)
            # pass # Draw all the rectangles around the smile
        
        # Find all eyes in the face
        for (x__, y__, w__, h__)  in eyes:
                
                # Draw a rectangle around a smile
                cv2.rectangle(the_face, (x__, y__), (x__ + w__, y__ + h__), (50, 50, 200), 4)
        
    # Show the current frame
    cv2.imshow("Smile detector", frame)
    
    # Display 
    cv2.waitKey(1) #se recarga cada 1 milisegundo el frame "fake key pressed"

# Cleanup    
webcam.release()
cv2.destroyAllWindows()

print('Code completed')