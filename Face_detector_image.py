import cv2

from random import randrange

trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #looks like if the image is to small the code does not recognice the face

# Choose an image to detect faces in 
img = cv2.imread('the_three_amigos.jpg')

# Must convert it to grayscale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #it is need to write BGR not RGB

# Detect Faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
# print(face_coordinates)

# Draw rectangle
for (x, y, w, h) in face_coordinates:
    
    cv2.rectangle(img, (x, y), (x+w, y+h), (randrange(128), randrange(256), randrange(256)), 2) #(0, 255, 0) color of the rectangle, 2 thickness of it


# Display the image
cv2.imshow('Clever Programmer Face Detector', img)
cv2.waitKey() #once a key is clicked the window close

print("Code completed")


