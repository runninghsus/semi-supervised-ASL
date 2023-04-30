import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models


def main():
    # page title
    st.markdown(f" <h1 style='text-align: left; color: #67286D; font-size:30px; "
                f"font-family:Avenir; font-weight:normal'>"
                f"Upload extracted pose along with annotation files."
                f""
                f"</h1> "
                , unsafe_allow_html=True)
    st.divider()
    colL, colR = st.columns(2)
    colL_exp = colL.expander('Upload extracted 3D hand pose csv file', expanded=True)
    # upload files
    uploaded_poses = colL_exp.file_uploader('Hand pose files',
                                        accept_multiple_files=True,
                                        type=['csv'])
    colR_exp = colR.expander('Upload annotation binary table csv file', expanded=True)
    uploaded_labels = colR_exp.file_uploader('Label files',
                                         accept_multiple_files=True,
                                         type=['csv'])
    mp_hands = mp.solutions.hands
    try:
        data_dict = np.load(f'./features_labels.npy',
                            allow_pickle=True).item()
        # define the smaller between the two
        low_res = []
        # downsample to 128x128 for Conv Net
        for high_res in data_dict['features']:
            gray_image = cv2.cvtColor(high_res, cv2.COLOR_BGRA2BGR)
            low_res.append(cv2.resize(gray_image, (128, 128)))
        # 80/20 split for training and test set
        train_images, test_images, train_labels, test_labels = \
            train_test_split(low_res,
                             np.array(data_dict['labels']),
                             test_size=0.2,
                             random_state=2023)
        # Normalize pixel values to be between 0 and 1
        train_images, test_images = np.array(train_images) / 255.0, np.array(test_images) / 255.0
        # label names
        class_names = [
            'background',
            'a', 'b', 'c', 'd', 'e',
            'f', 'g', 'h', 'i', 'j',
            'k', 'l', 'm', 'n', 'o',
            'p', 'q', 'r', 's', 't',
            'u', 'v', 'w', 'x', 'y',
            'z'
        ]
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
    except:
        labels_list = []
        features_list = []
        start_button = st.button('start extracting')
        if start_button:
            for i in range(len(uploaded_poses)):
                df_pose = pd.read_csv(uploaded_poses[i],
                                      low_memory=False)
                df_label = pd.read_csv(uploaded_labels[i],
                                       low_memory=False)
                print(uploaded_poses[i].name, uploaded_labels[i].name)
                label_framerate = 1 / (df_label.iloc[1, 0] - df_label.iloc[0, 0])
                # identify the poses frames
                frames_of_interest = df_pose.iloc[:, 1]
                labels = []
                features = []
                my_bar = st.progress(0)
                with st.spinner('extracting labels...'):
                    # extract labels from binary table
                    for i in frames_of_interest / (label_framerate * 3):
                        # add 1 to differentiate from unlabeled
                        try:
                            if np.max(np.array(df_label.loc[df_label['time'] == i])[0][1:]) > 0:
                                labels.append(
                                    np.argmax(
                                        np.array(df_label.loc[df_label['time'] == i])[0][1:], axis=0
                                    ) + 1
                                )

                            else:
                                labels.append(
                                    np.argmax(
                                        np.array(df_label.loc[df_label['time'] == i])[0][1:], axis=0
                                    )
                                )

                        except:
                            pass
                with st.spinner('extracting features...'):
                    # extract images
                    for row in np.arange(0, len(df_pose), 1):
                        if df_pose.iloc[row, 1] % 3 == 0:
                            fig = plt.figure(figsize=(4, 4))
                            ax = fig.add_subplot(projection='3d')
                            ax.view_init(elev=10, azim=10)
                            plotted_landmarks = {}
                            for i in range(21):
                                pose_digit = np.array(df_pose.iloc[row, (3 * i + 2):(3 * i + 2) + 3])
                                ax.scatter3D(
                                    xs=[-pose_digit[2]],
                                    ys=[pose_digit[0]],
                                    zs=[-pose_digit[1]],
                                    color='r',
                                    s=5,
                                    linewidth=1)
                                plotted_landmarks[i] = (-pose_digit[2],
                                                        pose_digit[0],
                                                        -pose_digit[1])

                            # Draws the connections if the start and end landmarks are both visible.
                            for connection in mp_hands.HAND_CONNECTIONS:
                                start_idx = connection[0]
                                end_idx = connection[1]
                                if start_idx in plotted_landmarks and end_idx in plotted_landmarks:
                                    landmark_pair = [
                                        plotted_landmarks[start_idx],
                                        plotted_landmarks[end_idx]
                                    ]
                                ax.plot3D(
                                    xs=[landmark_pair[0][0],
                                        landmark_pair[1][0]],
                                    ys=[landmark_pair[0][1],
                                        landmark_pair[1][1]],
                                    zs=[landmark_pair[0][2],
                                        landmark_pair[1][2]],
                                    color='k',
                                    linewidth=0.8)
                            ax.set_xticklabels('')
                            ax.set_yticklabels('')
                            ax.set_zticklabels('')
                            ax.axis('off')
                            fig.canvas.draw()
                            X = np.array(fig.canvas.renderer.buffer_rgba())
                            features.append(X)
                        my_bar.progress((row + 1) / len(df_pose))
                st.write(len(features), len(labels))
                features_list.append(features)
                labels_list.append(labels)
            all_features = np.vstack(features_list)
            all_labels = np.hstack(labels_list)
            print(len(all_features), all_labels.shape)
            data_dict = {'features': all_features, 'labels': all_labels}
            filename = fr'./features_labels'
            # save both npy and mat
            np.save(str.join('', (filename, '.npy')), data_dict)

    num_training_epoch = st.slider('Number of epochs?', 0, 100, 20)
    # _, mid_col, _ = st.columns([0.4, 1, 0.4])
    if st.button('start training a neural net'):
        # model architecture
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
        # start training
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        history = model.fit(train_images, train_labels, epochs=num_training_epoch,
                            validation_data=(test_images, test_labels))
        # plot results
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


    bottom_cont = st.container()
    with bottom_cont:
        st.divider()
        st.markdown(f" <h1 style='text-align: left; color: gray; font-size:16px; "
                    f"font-family:Avenir; font-weight:normal'>"
                    f"SignWave is developed by Alexander Hsu and Lucia Fang</h1> "
                    , unsafe_allow_html=True)