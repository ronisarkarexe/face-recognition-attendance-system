import cv2
import face_recognition
import os
import pandas as pd
import pickle

# Load the images and encode the faces
known_faces = []
known_names = []
for file in os.listdir("photos"):
    image = face_recognition.load_image_file("photos/" + file)
    encoding = face_recognition.face_encodings(image)[0]
    known_faces.append(encoding)
    known_names.append(os.path.splitext(file)[0])

# Save the encodings to disk
with open("encodings.pickle", "wb") as f:
    pickle.dump((known_faces, known_names), f)

# Load the encodings and attendance data from disk
try:
    df = pd.read_csv("attendance.csv")
except FileNotFoundError:
    df = pd.DataFrame(columns=["Name", "Present"])

# Open the camera
cap = cv2.VideoCapture(0)

while True:
    # Capture an image
    ret, frame = cap.read()

    # Find the faces in the image
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Loop through each face and see if it matches a known face
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_faces, face_encoding)

        # If a match is found, mark attendance
        if True in matches:
            index = matches.index(True)
            name = known_names[index]
            if name not in df["Name"].values:
                df = df._append({"Name": name, "Present": 1}, ignore_index=True)
                df.to_csv("attendance.csv", index=False)
                print(f"{name} is present!")

    # Display the image
    cv2.imshow("Attendance System", frame)

    # Wait for the user to press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the camera
cap.release()

# Close the window
cv2.destroyAllWindows()
