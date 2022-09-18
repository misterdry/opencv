import cv2

# Our images
img_file = 'Car image.jpg'

# Our pre-trained images classifier
classifier_file = 'car_detector.xml'

# Create opencv image
img = cv2.imread(img_file)

# Convert to grayscale (needed for harr cascade)
black_n_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Create a car classifier
car_tracker = cv2.CascadeClassifier(classifier_file)

# Detect Cars
cars = car_tracker.detectMultiScale(black_n_white)

# Draw rectangles around cars
for (x, y, w, h) in cars:
    
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the image with the faces spotted
cv2.imshow('Mistedry Car Detector', img)

# Don't autoclose (wait here in the code and listen for a key press)
cv2.waitKey()

print("Code completed")

