"""
Captures a single frame from the laptop's webcam - press SPACE to
save it, ESC to cancel. Lets you test the pipeline on a real photo
without a camera module on either board yet.
"""

import cv2

camera = cv2.VideoCapture(0)
print("Press SPACE to capture, ESC to cancel.")

while True:
    ok, frame = camera.read()
    if not ok:
        print("Could not read from webcam")
        break

    cv2.imshow("Webcam - SPACE to capture", frame)
    key = cv2.waitKey(1)

    if key % 256 == 27:  # ESC
        print("Cancelled")
        break
    elif key % 256 == 32:  # SPACE
        cv2.imwrite("webcam_capture.jpg", frame)
        print("Saved webcam_capture.jpg")
        break

camera.release()
cv2.destroyAllWindows()