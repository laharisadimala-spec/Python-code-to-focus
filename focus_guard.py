import cv2
import os
import time

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml'
)

cap = cv2.VideoCapture(0)

last_focus_time = time.time()
video_played = False
distraction_limit = 3


def play_warning_video():
    os.startfile("warning.mp4")


while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # better detection settings
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    distracted = True

    for (x, y, w, h) in faces:

        # face box
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.2, minNeighbors=5)

        # if two eyes detected -> focused
        if len(eyes) >= 2:
            distracted = False

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), (255,0,0), 2)


    if distracted:

        cv2.putText(frame, "DISTRACTED", (30,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

        if time.time() - last_focus_time > distraction_limit and not video_played:

            print("Distraction detected!")
            play_warning_video()

            video_played = True
            last_focus_time = time.time()

    else:

        cv2.putText(frame, "FOCUSED", (30,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)

        last_focus_time = time.time()
        video_played = False


    cv2.imshow("FocusGuard Eye Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
cv2.destroyAllWindows()