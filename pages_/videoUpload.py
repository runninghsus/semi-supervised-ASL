import streamlit as st
import cv2
import mediapipe as mp
import io
import pandas as pd
import numpy as np


def load_view():
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands
    st.subheader("Upload desired movie file and corresponding labels")
    colL, colR = st.columns(2)
    uploaded_movie = colL.file_uploader('Video file', type=['mp4'])
    temporary_location = False
    uploaded_labels = colR.file_uploader('Label file', type=['csv'])

    if uploaded_movie is not None:
        g = io.BytesIO(uploaded_movie.read())  # BytesIO Object
        temporary_location = 'temp_video.mp4'
        with open(temporary_location, 'wb') as out:  # Open temporary file as bytes
            out.write(g.read())  # Read bytes into file
        out.close()

    @st.cache(allow_output_mutation=True)
    def get_cap(location):
        print("Loading in function", str(location))
        video_stream = cv2.VideoCapture(str(location))
        total = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))
        # Check if camera opened successfully
        if video_stream.isOpened() == False:
            print("Error opening video  file")
        return video_stream, total

    scale_ = colL.slider('resolution scale', min_value=0.0, max_value=1.0, value=0.5)
    scaling_factorx = scale_
    scaling_factory = scale_
    image_placeholder = st.empty()

    idx = 0
    my_bar = st.progress(0)
    hand_pose = np.zeros((21, 3))

    # hand_poses = []
    hand_poses_list = []

    if temporary_location:
        while True:
            # here it is a CV2 object
            video_stream, total_frames = get_cap(temporary_location)
            ret, image = video_stream.read()

            with mp_hands.Hands(
                    static_image_mode=True,
                    max_num_hands=10,
                    min_detection_confidence=0.1) as hands:
                if ret:
                    idx += 1
                    image = cv2.resize(image, None, fx=scaling_factorx, fy=scaling_factory,
                                       interpolation=cv2.INTER_AREA)

                    # Read an image, flip it around y-axis for correct handedness output (see
                    # above).
                    # image = cv2.flip(cv2.imread(file), 1)
                    # Convert the BGR image to RGB before processing.
                    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    # results = hands.process(image)

                    # Print handedness and draw hand landmarks on the image.
                    # print('Handedness:', results.multi_handedness)
                    if not results.multi_hand_landmarks:
                        continue
                    image_height, image_width, _ = image.shape
                    annotated_image = image.copy()
                    for hand_landmarks in results.multi_hand_landmarks:

                        for idx_hand, landmark in enumerate(hand_landmarks.landmark):
                            hand_pose[idx_hand, :]=[landmark.x, landmark.y, landmark.z]
                        # tag frame number
                        hand_poses = np.hstack([idx, np.hstack(hand_pose)])
                        hand_poses_list.append(hand_poses)

                        # st.write('hand_landmarks:', hand_landmarks)
                        # print(
                        #     f'Index finger tip coordinates: (',
                        #     f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
                        #     f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
                        # )
                        mp_drawing.draw_landmarks(
                            annotated_image,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style())

                        if not hand_landmarks:
                            return

                        # plotted_landmarks = {}

                        # st.write(np.vstack(hand_poses_list).shape)

                            # print(idx_hand, landmark)
                            # print(landmark.x, landmark.y, landmark.z)
                    # cv2.imwrite(
                    #     '/tmp/annotated_image' + str(idx) + '.png', cv2.flip(annotated_image, 1)) #change this, and update str(idx)
                    # Draw hand world landmarks.
                    # if not results.multi_hand_world_landmarks:
                    #     continue
                    # for hand_world_landmarks in results.multi_hand_world_landmarks:
                    #     mp_drawing.plot_landmarks(
                    #         hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)

                else:
                    print("there was a problem or video was finished")
                    cv2.destroyAllWindows()
                    video_stream.release()
                    break
                # check if frame is None
                if image is None:
                    print("there was a problem None")
                    # if True break the infinite loop
                    break

            image_placeholder.image(annotated_image, channels="BGR", use_column_width=True)
            # update progress
            my_bar.progress(idx / total_frames)

            cv2.destroyAllWindows()
        hand_df = pd.DataFrame(data=np.vstack(hand_poses_list))
        hand_df.to_csv('./temp_video_pose.csv')
        video_stream.release()

        cv2.destroyAllWindows()
