import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

# Initialize video capture
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2, detectionCon=0.8)

# Configuration
offset = 20
imgSize = 300
counter = 0
folder = "D:/code_space/sign language detection/thankyou"  # Using forward slashes for better compatibility

# Create folder if it doesn't exist
if not os.path.exists(folder):
    os.makedirs(folder)

while True:
    success, img = cap.read()
    if not success:
        continue
        
    # Find hands in the image
    hands, img = detector.findHands(img, draw=True)  # Enable drawing by default
    
    # Process each detected hand
    for i, hand in enumerate(hands):
        # Get bounding box coordinates
        x, y, w, h = hand['bbox']
        
        # Create white background image
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        
        # Crop hand region with offset (ensure we don't go out of image bounds)
        y1, y2 = max(0, y-offset), min(img.shape[0], y + h + offset)
        x1, x2 = max(0, x-offset), min(img.shape[1], x + w + offset)
        imgCrop = img[y1:y2, x1:x2]
        
        # Skip if crop area is invalid
        if imgCrop.size == 0:
            continue
            
        imgCropShape = imgCrop.shape
        aspectRatio = h / w

        # Resize and center the hand image on white background
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        # Display each hand separately
        cv2.imshow(f'Hand {i+1}', imgWhite)
        
        # Save when 's' is pressed (each hand gets saved)
        key = cv2.waitKey(1)
        if key == ord("s"):
            counter += 1
            timestamp = int(time.time() * 1000)  # More precise timestamp
            hand_type = "left" if hand["type"] == "Left" else "right"
            cv2.imwrite(f'{folder}/Image_{hand_type}_{timestamp}.jpg', imgWhite)
            print(f"Saved {hand_type} hand image {counter}")

    # Display main image
    cv2.imshow('Image', img)
    
    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

