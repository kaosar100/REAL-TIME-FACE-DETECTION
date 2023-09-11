import threading as thd
import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counter = 0
face_match = False
ref_image = cv2.imread("My Photo.jpg") #put your image path here

cv2.imshow("check", ref_image)

cv2.waitKey(1)


def check_face(frame):
    global face_match
    print("in check_face")

    try:
        resp = DeepFace.verify(frame, ref_image.copy(),
                               model_name='Facenet', enforce_detection=False)

        if resp['verified']:
            face_match = True
            print("resp: ", face_match)

        else:
            face_match = False
            print("resp: ", face_match)

    except ValueError as e:
        face_match = False
        print("error from check", e)


while True:
    ret, frame = cap.read()

    if ret:
        if counter % 30 == 0:
            try:

                t1 = thd.Thread(target=check_face,
                                args=(frame.copy(), )).start()

            except ValueError as e:
                print("ERROR THREDING", e)

    if face_match:
        print("face matched: ", face_match)
        cv2.putText(frame, "Face Matched!", (20, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    else:
        print("face not matched: ", face_match)
        cv2.putText(frame, "NO Face Matched!", (20, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

    cv2.imshow("Streaming", frame)

    counter += 1

    key = cv2.waitKey(1)

    if key == ord('q'):
        break
cv2.destroyAllWindows()
