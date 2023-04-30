# import dependencies
import streamlit as st
import cv2
import mediapipe as mp
import io
import pandas as pd
import numpy as np


@st.cache_resource
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
    st.markdown(f" <h1 style='text-align: left; color: #67286D; font-size:30px; "
                f"font-family:Avenir; font-weight:normal'>"
                f"Upload videos, 3D hand pose will be extracted."
                f""
                f"</h1> "
                , unsafe_allow_html=True)
    st.divider()
    colL, colR = st.columns(2)
    colL_exp = colL.expander('Upload sign language videos', expanded=True)
    # your collection of uploaded movies
    uploaded_movies = colL_exp.file_uploader('Video file',
                                             accept_multiple_files=True,
                                             type=['mp4', 'avi'])
    # video scale, if 1, the resolution doesn't change,
    # if 0.5, resolution gets 1/2
    scale_ = colL_exp.slider('resolution scale',
                             min_value=0.0,
                             max_value=1.0,
                             value=0.5)
    # a button
    start_button = colL_exp.button('Start extracting hand pose')
    if start_button:
        # loops through your movie list
        for f, uploaded_movie in enumerate(uploaded_movies):
            # creates a spinner, spins until iteration is done
            with st.spinner(f'extracting video #{f}'):
                # image placeholder
                colR_exp = colR.expander('', expanded=True)
                image_placeholder = colR_exp.empty()
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
                my_bar = colR_exp.progress(0)
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
                                    mp_drawing.draw_landmarks(
                                        annotated_image,
                                        hand_landmarks,
                                        mp_hands.HAND_CONNECTIONS,
                                        mp_drawing_styles.get_default_hand_landmarks_style(),
                                        mp_drawing_styles.get_default_hand_connections_style())
                                    if not hand_landmarks:
                                        return
                            else:
                                print("there was a problem or video was finished")
                                cv2.destroyAllWindows()
                                video_stream.release()
                                break
                            # check if frame is None
                            if image is None:
                                print("there was a problem")
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

    bottom_cont = st.container()
    with bottom_cont:
        st.divider()
        st.markdown(f" <h1 style='text-align: left; color: gray; font-size:16px; "
                    f"font-family:Avenir; font-weight:normal'>"
                    f"SignWave is developed by Alexander Hsu and Lucia Fang</h1> "
                    , unsafe_allow_html=True)