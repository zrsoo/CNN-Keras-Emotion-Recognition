import cv2
import numpy
from keras.models import model_from_json

emotions = {0: "angry", 1: "disgusted", 2: "fearful", 3: "happy", 4: "neutral", 5: "sad", 6: "surprised"}

# Load the json file and create the model
file = open('emodel.json', 'r')
loaded_model = file.read()
file.close()
emodel = model_from_json(loaded_model)

# Get the weights
emodel.load_weights('emodel.h5')
print("The model has been prepared.")

# Begin recording
record = cv2.VideoCapture(0)

while True:
    ret, frame = record.read()
    frame = cv2.resize(frame, (1280, 720))

    if not ret:
        break

    figure_detect = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')

    # Convert image to grayscale because model is trained as such.
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect the faces in feed.
    faces = figure_detect.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # For each face, preprocess it.
    # x, y - left upper corner of rectangle containing face
    # w, h - width, height
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (0, 255, 0), 4)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]

        # Resize image to 48 by 48
        cropped_image = numpy.expand_dims(numpy.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        # Predict emotion.
        emotion = emodel.predict(cropped_image)

        # Get the maximum probability prediction index.
        maxindex = int(numpy.argmax(emotion))

        # Write prediction to image.
        cv2.putText(frame, emotions[maxindex], (x + 5, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('AI-ER', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

record.release()
cv2.destroyAllWindows()