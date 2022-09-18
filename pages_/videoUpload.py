import streamlit as st
import cv2
import mediapipe as mp
import io

# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
# mp_hands = mp.solutions.hands

# # # For static images:
# # IMAGE_FILES = []
# # with mp_hands.Hands(
# #     static_image_mode=True,
# #     max_num_hands=2,
# #     min_detection_confidence=0.5) as hands:
# #   for idx, file in enumerate(IMAGE_FILES):
# #     # Read an image, flip it around y-axis for correct handedness output (see
# #     # above).
# #     image = cv2.flip(cv2.imread(file), 1)
# #     # Convert the BGR image to RGB before processing.
# #     results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# #     # Print handedness and draw hand landmarks on the image.
# #     print('Handedness:', results.multi_handedness)
# #     if not results.multi_hand_landmarks:
# #       continue
# #     image_height, image_width, _ = image.shape
# #     annotated_image = image.copy()
# #     for hand_landmarks in results.multi_hand_landmarks:
# #       print('hand_landmarks:', hand_landmarks)
# #       print(
# #           f'Index finger tip coordinates: (',
# #           f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
# #           f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
# #       )
# #       mp_drawing.draw_landmarks(
# #           annotated_image,
# #           hand_landmarks,
# #           mp_hands.HAND_CONNECTIONS,
# #           mp_drawing_styles.get_default_hand_landmarks_style(),
# #           mp_drawing_styles.get_default_hand_connections_style())
# #     cv2.imwrite(
# #         '/tmp/annotated_image' + str(idx) + '.png', cv2.flip(annotated_image, 1))
# #     # Draw hand world landmarks.
# #     if not results.multi_hand_world_landmarks:
# #       continue
# #     for hand_world_landmarks in results.multi_hand_world_landmarks:
# #       mp_drawing.plot_landmarks(
# #         hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)

# # For webcam input:
# cap = cv2.VideoCapture(0)
# with mp_hands.Hands(
#     model_complexity=0,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5) as hands:
#   while cap.isOpened():
#     success, image = cap.read()
#     if not success:
#       print("Ignoring empty camera frame.")
#       # If loading a video, use 'break' instead of 'continue'.
#       continue

#     # To improve performance, optionally mark the image as not writeable to
#     # pass by reference.
#     image.flags.writeable = False
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     results = hands.process(image)

#     # Draw the hand annotations on the image.
#     image.flags.writeable = True
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     if results.multi_hand_landmarks:
#       for hand_landmarks in results.multi_hand_landmarks:
#         mp_drawing.draw_landmarks(
#             image,
#             hand_landmarks,
#             mp_hands.HAND_CONNECTIONS,
#             mp_drawing_styles.get_default_hand_landmarks_style(),
#             mp_drawing_styles.get_default_hand_connections_style())
#     # Flip the image horizontally for a selfie-view display.
#     cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
#     if cv2.waitKey(5) & 0xFF == 27:
#       break
# cap.release()

# def get_default_hand_landmarks_style() -> Mapping[int, DrawingSpec]:
#   """Returns the default hand landmarks drawing style.
#
#   Returns:
#       A mapping from each hand landmark to its default drawing spec.
#   """
#   hand_landmark_style = {}
#   for k, v in _HAND_LANDMARK_STYLE.items():
#     for landmark in k:
#       hand_landmark_style[landmark] = v
#   return hand_landmark_style

# 1. Collect images for deep learning using your webcam and OpenCV
# 2. Label images for sign language detection using LabelImg
# 3. Setup Tensorflow Object Detection pipeline configuration
# 4. Use transfer learning to train a deep learning model
# 5. Detect sign language in real time using OpenCV
# https://www.youtube.com/watch?v=pDXdlXlaCco 

# def load_view():
#     st.title('Video Upload')

def load_view():
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands
    # st.markdown("# Video Upload")
    # st.sidebar.markdown("# Video Upload")
    # st.video("https://www.youtube.com/watch?v=eeAq4gkOEUY")


    st.title("Play Uploaded File")

    uploaded_file = st.file_uploader("Choose a video...", type=["mp4"])

    temporary_location = False

    if uploaded_file is not None:
        g = io.BytesIO(uploaded_file.read())  ## BytesIO Object
        temporary_location = "testout_simple.mp4"

        with open(temporary_location, 'wb') as out:  ## Open temporary file as bytes
            out.write(g.read())  ## Read bytes into file

        # close file
        out.close()


    @st.cache(allow_output_mutation=True)
    def get_cap(location):
        print("Loading in function", str(location))
        video_stream = cv2.VideoCapture(str(location))
        total = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))
        # st.write(total)
        # Check if camera opened successfully
        if (video_stream.isOpened() == False):
            print("Error opening video  file")
        return video_stream, total


    scaling_factorx = 0.75
    scaling_factory = 0.75
    image_placeholder = st.empty()

    idx = 0
    my_bar = st.progress(0)
    if temporary_location:
        while True:
            # here it is a CV2 object
            video_stream, total_frames = get_cap(temporary_location)
            # st.write(video_stream)
            # video_stream = video_stream.read()
            ret, image = video_stream.read()
            with mp_hands.Hands(
                static_image_mode=True,
                max_num_hands=10,
                min_detection_confidence=0.1) as hands:
              if ret:
                  idx += 1
                  image = cv2.resize(image, None, fx=scaling_factorx, fy=scaling_factory, interpolation=cv2.INTER_AREA)

          
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
                  # cv2.imwrite(
                  #     '/tmp/annotated_image' + str(idx) + '.png', cv2.flip(annotated_image, 1)) #change this, and update str(idx)
                  # Draw hand world landmarks.
                  if not results.multi_hand_world_landmarks:
                    continue
                  for hand_world_landmarks in results.multi_hand_world_landmarks:
                    mp_drawing.plot_landmarks(
                      hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)

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
            my_bar.progress(idx/total_frames)

            cv2.destroyAllWindows()
        video_stream.release()


        cv2.destroyAllWindows()



    # for uploaded_file in uploaded_files:
    #     # bytes_data = uploaded_file.read()
    #     st.write(uploaded_file)
    #     video_file = open(uploaded_file, 'rb')
    #     video_bytes = video_file.read()
    #     st.write("filename:", uploaded_file.name)
    #     st.video(video_bytes)
