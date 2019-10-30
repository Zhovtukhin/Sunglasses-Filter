import cv2
import numpy as np
import dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


glasses =   cv2.imread("sunglass.png", -1)
glasses = glasses[ 36:256, 20:610]
drops =   cv2.imread("drops.jpg", 1)
drops = drops[:glasses.shape[0],:glasses.shape[1]]
glasses[:,:,:3] = cv2.addWeighted(glasses[:,:,:3], 0.4, drops, 0.6, 0)

source = 0

cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
srcPoints = np.array([[0, 84], [589, 84], [115, 0], [460, 0], [293, 37], [115, 219], [460, 219], [255, 64], [350, 64]], dtype=float)

#vid_writer = cv2.VideoWriter('output-dnn-{}.avi'.format(str(source).split(".")[0]),cv2.VideoWriter_fourcc('M','J','P','G'), 15, (frame.shape[1],frame.shape[0]))

frame_count = 0
while(1):
    hasFrame, frame = cap.read()
    if not hasFrame:
        break
    frame_count += 1

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        dstPoints = []

        landmarks = predictor(gray, face)

        for n in [0, 16, 19, 24, 27, 29, 39, 42]:
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            if n == 29:
                dstPoints.append([dstPoints[-3][0], y])
                dstPoints.append([dstPoints[-3][0], y])
                #cv2.circle(frame, (dstPoints[-1][0], y), 3, (255, 0, 0), -1)
                #cv2.circle(frame, (dstPoints[-2][0], y), 3, (255, 0, 0), -1)
            else:
                dstPoints.append([x, y])
                #cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)

        if (srcPoints.shape== np.array(dstPoints) .shape):
            h, status = cv2.findHomography(srcPoints, np.array(dstPoints) )
            imH = cv2.warpPerspective(glasses, h, (frame.shape[1], frame.shape[0]))
            glassMask = (np.round(cv2.merge((imH[:,:,3],imH[:,:,3],imH[:,:,3]))/255)).astype(np.uint8)
            sunglasses = cv2.addWeighted(frame, 0.4, imH[:, :, :3], 0.6, 0)
            frame = cv2.add(cv2.multiply(frame, (1 - glassMask)), cv2.multiply(sunglasses, glassMask))

    cv2.imshow("Sunglasses", frame)
    k = cv2.waitKey(10)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
