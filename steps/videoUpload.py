# import dependencies
import streamlit as st
import cv2
import mediapipe as mp
import io
import pandas as pd
import numpy as np
# creating multipage app
from hands import swap_app

import categories

CATEGORY = categories.VIDEO_UPLOAD
TITLE = "Video Upload"


@st.cache(allow_output_mutation=True)
def get_cap(location):
    print("Loading in function", str(location))
    video_stream = cv2.VideoCapture(str(location))
    total = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))
    # Check if camera opened successfully
    if video_stream.isOpened() == False:
        print("Error opening video  file")
    return video_stream, total


def main():
    # how should I draw the dots on the hand, colors, size
    mp_drawing = mp.solutions.drawing_utils
    # how do I connect the dots to make a hand, lines between dots
    mp_drawing_styles = mp.solutions.drawing_styles
    # main prediction classifier, where in image is the hand
    mp_hands = mp.solutions.hands

    # page title
    st.subheader("Hand Tracking")
    # your collection of uploaded movies
    uploaded_movies = st.file_uploader('Video file',
                                       accept_multiple_files=True,
                                       type=['mp4', 'avi'])
    # video scale, if 1, the resolution doesn't change,
    # if 0.5, resolution gets 1/2
    scale_ = st.slider('resolution scale',
                       min_value=0.0,
                       max_value=1.0,
                       value=0.5)
    # a button
    start_button = st.button('Start extracting hand pose')
    if start_button:
        # loops through your movie list
        for f, uploaded_movie in enumerate(uploaded_movies):
            # creates a spinner, spins until iteration is done
            with st.spinner(f'extracting video #{f}'):
                # image placeholder
                image_placeholder = st.empty()
                temporary_location = False
                if uploaded_movie is not None:
                    g = io.BytesIO(uploaded_movie.read())  # BytesIO Object
                    temporary_location = f'temp_video_{f}.mp4'
                    with open(temporary_location, 'wb') as out:  # Open temporary file as bytes
                        out.write(g.read())  # Read bytes into file
                    out.close()
                # frame counter
                idx = 0
                # progress bar, 0 to 1
                my_bar = st.progress(0)
                # initialization of an array, number of dots on hand x 3 (x, y, z)
                # mediapipe default output is 21x3
                hand_pose = np.zeros((21, 3))
                # list that stores all the frames
                hand_poses_list = []

                if temporary_location:
                    while True:
                        # here it is a CV2 object
                        video_stream, total_frames = get_cap(temporary_location)
                        ret, image = video_stream.read()
                        # classifier params set, and call is as hands
                        with mp_hands.Hands(
                                static_image_mode=True,
                                max_num_hands=10,
                                min_detection_confidence=0.1) as hands:
                            # if there is an image that is readable
                            if ret:
                                # adding one to image/frame counter
                                idx += 1
                                #
                                image = cv2.resize(image, None, fx=scale_, fy=scale_,
                                                   interpolation=cv2.INTER_AREA)

                                # Convert the BGR image to RGB before processing.
                                results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                                # if there are no hands
                                if not results.multi_hand_landmarks:
                                    continue
                                # limit to single hand
                                if len(results.multi_hand_landmarks) > 1:
                                    continue
                                # dimensions of the image
                                image_height, image_width, _ = image.shape
                                # making a copy, then we draw on it
                                # (not necessary here but this was prob good practice)
                                annotated_image = image.copy()

                                # iterate through all detected hands, I limited it to single hand now
                                for hand_landmarks in results.multi_hand_landmarks:
                                    # enumerating over the landmarks (21 of them)
                                    for idx_hand, landmark in enumerate(hand_landmarks.landmark):
                                        # store x,y,z coordinate by index of frame
                                        hand_pose[idx_hand, :] = [landmark.x, landmark.y, landmark.z]
                                    # tag frame number, as some frames possibly don't have hands
                                    hand_poses = np.hstack([idx, np.hstack(hand_pose)])
                                    # store into a list of all time points/frames, if single hand
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
                    hand_df.to_csv(f'./temp_video_{f}_pose.csv')
                    video_stream.release()
                    cv2.destroyAllWindows()
