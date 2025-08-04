import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Initialize modules
detector = HandDetector(maxHands=1)
try:
    classifier = Classifier(
       "D:\code_space\sign language detection\converted_keras\keras_model.h5",
        "D:\code_space\sign language detection\converted_keras\labels.txt"
    )
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Constants
offset = 20
imgSize = 300
labels = ["Hello", "Thank you", "Yes",]

while True:
    try:
        success, img = cap.read()
        if not success:
            print("Failed to capture image")
            break

        imgOutput = img.copy()
        hands, img = detector.findHands(img)
        
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            # Create white background image
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            
            # Crop hand region with safety check
            imgCrop = img[max(0, y-offset):y + h + offset, max(0, x-offset):x + w + offset]
            if imgCrop.size == 0:
                continue

            aspectRatio = h / w

            if aspectRatio > 1:
                # Vertical hand
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                # Horizontal hand
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            # Get prediction
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            print(f"Prediction: {labels[index]} (Confidence: {prediction[index]:.2f})")

            # Display results
            cv2.rectangle(imgOutput, (x-offset, y-offset-70), 
                         (x-offset+400, y-offset+60-50), (0, 255, 0), cv2.FILLED)
            cv2.putText(imgOutput, labels[index], (x, y-30), 
                        cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
            cv2.rectangle(imgOutput, (x-offset, y-offset), 
                         (x + w + offset, y + h + offset), (0, 255, 0), 4)

            cv2.imshow('Hand Crop', imgCrop)
            cv2.imshow('White Background', imgWhite)

        cv2.imshow('Sign Language Detection', imgOutput)
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print(f"Error: {e}")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()


