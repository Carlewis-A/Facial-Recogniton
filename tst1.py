# Name: Carlewis Chambang Akana
# Department: Computer engineering
# Subject: Artificial Intelligence

import cv2
# import sys

# Get user supplied values
# imagePath = sys.argv[0]
# imagePath = "C:/Users/Geradine/3D Objects/Programming/python/face_detection/gig-ekonomisi.jpeg"
imagePath = "gig-ekonomisi.jpeg"
cascPath = "haarcascade_frontalface_default.xml"

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=2.1,
    minNeighbors=5,
    minSize=(30, 30),
    # flags = cv2.cv.CV_HAAR_SCALE_IMAGE
    flags=cv2.CASCADE_SCALE_IMAGE
)

print("Found {0} faces!".format(len(faces)))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Faces found", image)
cv2.waitKey(0)