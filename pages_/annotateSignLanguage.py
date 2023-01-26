import mediapipe as mp
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib import animation
import streamlit.components.v1 as components
from matplotlib import rc
# plt.rcParams['animation.ffmpeg_path'] = r'C:\Users\alexa\Anaconda3\pkgs\ffmpeg-4.3.1-ha925a31_0\Library\bin\ffmpeg.exe'
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2
from PIL import Image

rc('animation', html='jshtml')


def animation_create(image_list):
    fig = plt.figure()
    #     plt.axis('off')
    im = plt.imshow(image_list[0])

    def animate(k):
        im.set_array(image_list[k])
        return im

    ani = animation.FuncAnimation(fig, animate, frames=len(image_list), blit=False)
    return ani


def load_view():
    st.subheader("# Annotate Sign Language")
    # st.subheader("Upload desired movie file and corresponding labels")
    colL, colR = st.columns(2)
    uploaded_pose = colL.file_uploader('Hand pose file', type=['csv'])
    uploaded_labels = colR.file_uploader('Label file', type=['csv'])

    mp_hands = mp.solutions.hands
    # try:
    data_dict = np.load('./features_labels.npy', allow_pickle=True).item()
    low_res = []
    for high_res in data_dict['features']:
        gray_image = cv2.cvtColor(high_res, cv2.COLOR_BGRA2BGR)
        low_res.append(cv2.resize(gray_image, (128, 128)))

    train_images, test_images, train_labels, test_labels = train_test_split(low_res,
                                                                            np.array(data_dict['labels']),
                                                                            test_size=0.2, random_state=42)
    # (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    # st.write(train_images.shape)
    # Normalize pixel values to be between 0 and 1
    train_images, test_images = np.array(train_images) / 255.0, np.array(test_images) / 255.0
    class_names = ['background', 'a', 'b', 'c', 'd', 'e',
                   'f', 'g', 'h', 'i', 'j',
                   'k', 'l', 'm', 'n', 'o',
                   'p', 'q', 'r', 's', 't',
                   'u', 'v', 'w', 'x', 'y',
                   'z'
                   ]
    # train_images_ds = cv2.resize(train_images[0], (32, 32, 4))
    # st.write(train_images_ds.shape)

    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i])
        # The CIFAR labels happen to be arrays,
        # which is why you need the extra index
        plt.title(f'Label: {class_names[train_labels[i]]}')
    plt.show()
    st.pyplot()
    num_training_epoch = st.slider('Number of epochs?', 0, 100, 20)
    image = Image.open('./assets/images/conv_net_example.png')
    st.image(image)
    if st.button('start training a neural net'):

        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.summary()

        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(27))
        model.summary()

        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        history = model.fit(train_images, train_labels, epochs=num_training_epoch,
                            validation_data=(test_images, test_labels))

        fig, ax = plt.subplots(1, 1, figsize=(11, 8))
        ax.plot(history.history['accuracy'], label='accuracy', color='indianred')
        ax.plot(history.history['val_accuracy'], label='val_accuracy', color='dodgerblue')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_ylim([0.5, 1])
        ax.legend(loc='lower right')

        test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
        st.success(f'Accuracy of {test_acc}!')
        st.pyplot(fig)

# st.write(data_dict['labels'], data_dict['features'][0])
#     fig, ax = plt.subplots(1, 1)
#     ax.imshow(data_dict['features'][19])
#     ax.set_title(data_dict['labels'][19])
#     st.pyplot(fig)

# ani = animation_create(data_dict['features'])

# components.html(ani.to_jshtml(), height=800)
# return ani

# except:
#     df = pd.read_csv(uploaded_pose, low_memory=False)
#     df_labels = pd.read_csv(uploaded_labels, low_memory=False)
#
#     framerate=1/(df_labels.iloc[1, 0] - df_labels.iloc[0, 0])
#     frames_of_interest = df.iloc[:, 1]
#     labels = []
#     with st.spinner('extracting labels...'):
#         for i in frames_of_interest/framerate:
#             # add 1 to differentiate from unlabeled
#             labels.append(np.argmax(np.array(df_labels.loc[df_labels['time']==i])[0][1:], axis=0)+1)
#
#
#     # pose_digit = []
#     # for i in range(21):
#     #     pose_digit.append(df.iloc[:, (3 * i + 2):(3 * i + 2) + 2])
#
#     features = []
#     my_bar = st.progress(0)
#     with st.spinner('extracting features...'):
#         for row in range(len(df)):
#
#             fig = plt.figure(figsize=(4, 4))
#             ax = fig.add_subplot(projection='3d')
#
#             # # ax = fig.add_subplot(rows, cols, m + 1, projection='3d')
#             ax.view_init(elev=10, azim=10)
#             plotted_landmarks = {}
#             for i in range(21):
#                 pose_digit = np.array(df.iloc[row, (3 * i + 2):(3 * i + 2) + 3])
#                 ax.scatter3D(
#                     xs=[-pose_digit[2]],
#                     ys=[pose_digit[0]],
#                     zs=[-pose_digit[1]],
#                     color='r',
#                     s=5,
#                     linewidth=1)
#                 plotted_landmarks[i] = (-pose_digit[2],
#                                         pose_digit[0],
#                                         -pose_digit[1])
#
#             # Draws the connections if the start and end landmarks are both visible.
#             for connection in mp_hands.HAND_CONNECTIONS:
#                 start_idx = connection[0]
#                 end_idx = connection[1]
#
#                 if start_idx in plotted_landmarks and end_idx in plotted_landmarks:
#                     landmark_pair = [
#                         plotted_landmarks[start_idx], plotted_landmarks[end_idx]
#                     ]
#                 ax.plot3D(
#                     xs=[landmark_pair[0][0], landmark_pair[1][0]],
#                     ys=[landmark_pair[0][1], landmark_pair[1][1]],
#                     zs=[landmark_pair[0][2], landmark_pair[1][2]],
#                     color='k',
#                     linewidth=0.8)
#             ax.set_xticklabels('')
#             ax.set_yticklabels('')
#             ax.set_zticklabels('')
#             ax.axis('off')
#
#             fig.canvas.draw()
#
#             X = np.array(fig.canvas.renderer.buffer_rgba())
#             features.append(X)
#
#             my_bar.progress((row+1) / len(df))
#
#         data_dict = {'features': features, 'labels': labels}
#
#         filename = r'./features_labels'
#         # save both npy and mat
#         np.save(str.join('', (filename, '.npy')), data_dict)


# data_train = [hand_pose_stacked]


# # subplot with various fingerspelling skeleton
# fig = plt.figure(figsize=(16, 16))
# num_landmarks = 21
# rows = 6
# cols = 4
#
# nskip = 50
# offset = 0
#
# for m in range(int(rows * cols)):
#     try:
#         ax = fig.add_subplot(rows, cols, m + 1, projection='3d')
#         ax.view_init(elev=10, azim=10)
#         plotted_landmarks = {}
#         for i in range(21):
#             pose_digit = np.array(df.iloc[:, (3 * i + 2):(3 * i + 2) + 3])
#             ax.scatter3D(
#                 xs=[-pose_digit[m * nskip + offset, 2]],
#                 ys=[pose_digit[m * nskip + offset, 0]],
#                 zs=[-pose_digit[m * nskip + offset, 1]],
#                 color='r',
#                 s=5,
#                 linewidth=1)
#             plotted_landmarks[i] = (-pose_digit[m * nskip + offset, 2],
#                                     pose_digit[m * nskip + offset, 0],
#                                     -pose_digit[m * nskip + offset, 1])
#
#         # Draws the connections if the start and end landmarks are both visible.
#         for connection in mp_hands.HAND_CONNECTIONS:
#             start_idx = connection[0]
#             end_idx = connection[1]
#
#             if start_idx in plotted_landmarks and end_idx in plotted_landmarks:
#                 landmark_pair = [
#                     plotted_landmarks[start_idx], plotted_landmarks[end_idx]
#                 ]
#             ax.plot3D(
#                 xs=[landmark_pair[0][0], landmark_pair[1][0]],
#                 ys=[landmark_pair[0][1], landmark_pair[1][1]],
#                 zs=[landmark_pair[0][2], landmark_pair[1][2]],
#                 color='k',
#                 linewidth=0.8)
#         ax.set_xticklabels('')
#         ax.set_yticklabels('')
#         ax.set_zticklabels('')
#         ax.axis('off')
#     except:
#         pass
# plt.show()
# plt.axis('off')
# st.pyplot(fig)
# except:
#     st.warning('please upload both files')

# st.write(uploaded_pose)
# st.sidebar.markdown("# Annotate Sign Language")
