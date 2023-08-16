import numpy as np
import cv2
import os
import sys

args: list[int] = sys.argv
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'
cap = cv2.VideoCapture(sys.argv[1])

fps = cap.get(cv2.CAP_PROP_FPS)
realTriggerTime = 2
triggerTime = int(realTriggerTime * 5 / 3)

if fps == 0:
    triggerTime = int(triggerTime * 20)
    realTriggerTime = int(realTriggerTime * 20)
else:
    triggerTime = int(triggerTime * fps)
    realTriggerTime = int(realTriggerTime * fps)

counter = int(triggerTime + 1)
tamperingCounter = 0
fpsCounter = 0
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

fourcc = cv2.VideoWriter_fourcc(*"XVID")
videoWriter = cv2.VideoWriter(f'output.avi', fourcc, 24, (frame_width, frame_height))
fgbg = cv2.createBackgroundSubtractorMOG2()
ret, frame = cap.read()
fgmask = fgbg.apply(frame)
kernel = np.ones((5, 5), np.uint8)
grey_fps = 5


def create_bar(height, width, color):
    bar = np.zeros((height, width, 3), np.uint8)
    bar[:] = color
    red, green, blue = int(color[2]), int(color[1]), int(color[0])
    return bar, (red, green, blue)

height, width, _ = np.shape(frame)

data = np.reshape(frame, (height * width, 3))
data = np.float32(data)
number_clusters = 5
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
flags = cv2.KMEANS_RANDOM_CENTERS
compactness, labels, centers = cv2.kmeans(data, number_clusters, None, criteria, 10, flags)


while True:
    ret, frame = cap.read()

    if ret:

        font = cv2.FONT_HERSHEY_SIMPLEX
        bars = []
        rgb_values = []

        for index, row in enumerate(centers):
            bar, rgb = create_bar(200, 200, row)
            bars.append(bar)
            rgb_values.append(rgb)
        img_bar = np.hstack(bars)

        for index, row in enumerate(rgb_values):
            image = cv2.putText(img_bar, f'{index + 1}. RGB: {row}', (5 + 200 * index, 200 - 10),
                                font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
            print(f'{index + 1}. RGB{row}')
            cv2.imshow('frame', frame)
            cv2.imshow('Dominant colors', img_bar)
        print(len(bars))

        if (counter % triggerTime) == 0:
            counter = 0
            print(fps)

            if tamperingCounter >= realTriggerTime:
                if fpsCounter >= tamperingCounter * grey_fps:

                    cv2.putText(frame, "Real Tampering", (3, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    print("real tampering")
            tamperingCounter = 0
        counter += 1
        a = 0
        bounding_rect = []
        fgmask = fgbg.apply(frame)
        fgmask = cv2.erode(fgmask, kernel, iterations=5)
        fgmask = cv2.dilate(fgmask, kernel, iterations=5)
        cv2.imshow('frame', frame)
        contours, _ = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.putText(frame, "fps: " + str(fps), (3, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        for i in range(0, len(contours)):
            bounding_rect.append(cv2.boundingRect(contours[i]))

            if bounding_rect[i][2] >= 40 or bounding_rect[i][3] >= 40:
                a = a + (bounding_rect[i][2]) * bounding_rect[i][3]

            if a >= int(frame.shape[0]) * int(frame.shape[1]) / 3:
                cv2.putText(frame, "TAMPERING DETECTED", (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                tamperingCounter += 1
                fpsCounter += fps
            cv2.imshow('frame', frame)
            videoWriter.write(frame)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
