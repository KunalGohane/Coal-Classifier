from ultralytics import YOLO

import numpy as np


model = YOLO('C:/Users/kunal/OneDrive/Desktop/local_env/local_env/runs/classify/train5/weights/last.pt')  # loading our model


results = model('C:/Users/Kaustubh Yewale/Desktop/1.jpg')  # predict on an image


names_dict = results[0].names
probs = results[0].probs.data.tolist()

print(names_dict)

print(probs)
print(names_dict[np.argmax(probs)])





















# import cv2
# import numpy as np
# from ultralytics import YOLO

# # Load the YOLO model
# model = YOLO('E:/image-classification-yolov8-main/local_env/runs/classify/train5/weights/last.pt')

# # Function to perform object detection on the captured image
# def detect_objects(image):
#     results = model(image)  # Perform object detection
#     return results

# # Open the camera
# cap = cv2.VideoCapture(1)

# while True:
#     # Capture frame-by-frame
#     ret, frame = cap.read()

#     # Display the captured frame
#     cv2.imshow('Camera', frame)

#     # Check if the user pressed 'q' to quit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#     # Check if the user pressed 'c' to capture an image
#     elif cv2.waitKey(1) & 0xFF == ord('c'):
#         # Save the captured image
#         cv2.imwrite('captured_image.jpg', frame)
        
#         # Perform object detection on the captured image
#         results = detect_objects(frame)
        
        
      

# # Release the camera and close all OpenCV windows
# cap.release()
# cv2.destroyAllWindows()
