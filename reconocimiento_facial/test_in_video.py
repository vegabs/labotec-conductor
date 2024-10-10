import face_recognition
import cv2
import numpy as np
import pickle


video_capture = cv2.VideoCapture(0)
process_this_frame = True

MODEL_FILENAME = "trained_knn_model_6.clf"
def predict(X_img_path, knn_clf=None, model_path=None, distance_threshold=0.6):
    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # Load image file and find face locations
    X_img = X_img_path
    X_face_locations = face_recognition.face_locations(X_img)

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []

    # Find encodings for faces in the test iamge
    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Only process every other frame of video to save time
    if process_this_frame:
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        # rgb_small_frame = small_frame[:, :, ::-1]
        code = cv2.COLOR_BGR2RGB
        rgb_small_frame = cv2.cvtColor(small_frame, code)
        
        # Find all people in the image using a trained classifier model
        # Note: You can pass in either a classifier file name or a classifier model instance
        predictions = predict(rgb_small_frame, model_path=MODEL_FILENAME)

        # Print results on the console
        for name, (top, right, bottom, left) in predictions:
            print("==== Found {} at ({}, {})".format(name, left, top))

    process_this_frame = not process_this_frame

    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()