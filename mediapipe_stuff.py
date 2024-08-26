import cv2
import mediapipe as mp
import warnings
import os
import csv

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

pose_folder = "./data_SSL/pose_landmarks/"
video_path = './data_SSL/videos/'

def save_results(writer_pose, writer_face, writer_left, writer_right, frame_count, results):
    face_landmarks = results.face_landmarks
    pose_landmarks = results.pose_landmarks
    left_hand = results.left_hand_landmarks
    right_hand = results.right_hand_landmarks

    if face_landmarks is not None:
        for idx, landmark in enumerate(face_landmarks.landmark):
            writer_face.writerow([
                frame_count, idx,
                landmark.x, landmark.y, landmark.z, landmark.visibility
            ])
    if pose_landmarks is not None:
        for idx, landmark in enumerate(pose_landmarks.landmark):
            writer_pose.writerow([
                frame_count, idx,
                landmark.x, landmark.y, landmark.z, landmark.visibility
            ])
    if left_hand is not None and right_hand is not None:
        for idx, landmark in enumerate(left_hand.landmark):
            writer_left.writerow([
                frame_count, idx,
                landmark.x, landmark.y, landmark.z, landmark.visibility
            ])
        for idx, landmark in enumerate(right_hand.landmark):
            writer_right.writerow([
                frame_count, idx,
                landmark.x, landmark.y, landmark.z, landmark.visibility
            ])


for filename in os.listdir(video_path):

    if not filename.endswith(".mp4"):
        continue

    if os.path.exists(pose_folder + filename[:-4] + "_face.csv"):
        print("file exists, skipping")
        continue

    print(filename)
    cap = cv2.VideoCapture(video_path+filename)
    pose_file_face = open(pose_folder + filename[:-4] + "_face.csv", mode='w', newline='')
    pose_file_pose = open(pose_folder + filename[:-4] + "_pose.csv", mode='w', newline='')
    pose_file_left = open(pose_folder + filename[:-4] + "_left.csv", mode='w', newline='')
    pose_file_right = open(pose_folder + filename[:-4] + "_right.csv", mode='w', newline='')

    writer_pose = csv.writer(pose_file_pose)
    writer_face = csv.writer(pose_file_face)
    writer_left = csv.writer(pose_file_left)
    writer_right = csv.writer(pose_file_right)

    frame_count = 0


    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:
      while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("cant read image")
            break

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)

        save_results(writer_pose, writer_face, writer_left, writer_right, frame_count, results)

        a = results.face_landmarks

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            image,
            results.face_landmarks,
            mp_holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_contours_style())
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles
            .get_default_pose_landmarks_style())

        if results.left_hand_landmarks is not None and\
            results.right_hand_landmarks is not None:
            mp_drawing.draw_landmarks(
                image,
                results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))
            mp_drawing.draw_landmarks(
                image,
                results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))
            # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break
        frame_count += 1

    cap.release()