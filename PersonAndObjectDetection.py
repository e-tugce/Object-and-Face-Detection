import cv2
import numpy as np
import tensorflow as tf
import json

model_path = r"C:\Users\Emel Tuğçe Kara\Downloads\ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8\ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8\saved_model"
model = tf.saved_model.load(model_path)
infer = model.signatures["serving_default"]

cascPath = r"C:\Users\TugceKara\haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

with open(r"C:\Users\Emel Tuğçe Kara\OneDrive\Resimler\Masaüstü\coco_labels.txt", 'r') as file:
    class_labels = [line.strip() for line in file.readlines()]

video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    input_tensor = tf.convert_to_tensor([rgb_frame], dtype=tf.uint8)
    detections = infer(input_tensor)

    boxes = detections['detection_boxes'][0].numpy()
    classes = detections['detection_classes'][0].numpy().astype(np.int32)
    scores = detections['detection_scores'][0].numpy()

    for i in range(len(boxes)):
        box = boxes[i]
        score = scores[i]
        class_id = classes[i]


        if class_id<len(class_labels):
            class_label = class_labels[class_id-1]
        else:
            class_label = 'Unknown'

        if score > 0.6:
            ymin, xmin, ymax, xmax = box
            xmin = int(xmin * frame.shape[1])
            xmax = int(xmax * frame.shape[1])
            ymin = int(ymin * frame.shape[0])
            ymax = int(ymax * frame.shape[0])

            label = f"{class_label}: {score:.2f}"

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
            cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.imshow('Video', frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

