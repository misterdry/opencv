import cv2

# Our video
# video = cv2.VideoCapture('Tesla video.mp4')
video = cv2.VideoCapture('Pedestrian video.mp4')

# Our pre-trained images classifier
car_tracker_file = 'car_detector.xml'
pedestrian_tracker_file = 'haarcascade_fullbody.xml'

# Create car classifier
car_tracker = cv2.CascadeClassifier(car_tracker_file)
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_tracker_file)

# Run forever until car stops or something
while True:
    
    # Read the current frame
    (read_succesful, frame) = video.read()
    
    # Safe coding
    if read_succesful:
        
        # Must convert to grayscale
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break
    
    # Create car classifier
    cars = car_tracker.detectMultiScale(grayscaled_frame)
    pedestrians = pedestrian_tracker.detectMultiScale(grayscaled_frame)
   
    # Draw rectangles around cars
    for (x, y, w, h) in cars:
       cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
   
    # Draw rectangles around cars
    for (x, y, w, h) in pedestrians:
       cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
   
    # Display the video with cars
    cv2.imshow('Misterdry Self driving car', frame)
    
    # Don't autoclose (wait here in the code and listen for a key press)
    key = cv2.waitKey(1)
    
    # Stop if q or Q is pressed
    if key==81 or key==113:
        break

# Release the VideoCapture object
video.release()
