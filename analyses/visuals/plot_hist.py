import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import mediapipe as mp


def main(filepath='../../temp_video_0_pose.csv'):
    df = pd.read_csv(filepath, low_memory=False)
    pose_digit1 = np.array(df.iloc[:, 1:4])
    pose_digit2 = np.array(df.iloc[:, 4:7])
    pose_digit3 = np.array(df.iloc[:, 7:10])
    rows = 1
    cols = 1
    x1, y1 = pose_digit1[:, 0], pose_digit1[:, 1]
    x2, y2 = pose_digit2[:, 0], pose_digit2[:, 1]
    data_dict = {'condition': np.hstack([np.repeat(i, len(x1)) for i in range(2)]),
                 'data': np.hstack((x1, x2))}
    data_df = pd.DataFrame(data=data_dict)
    # print(data_df)
    # fig is entire object to save,
    # ax is axis - you can design multiple axes on the same figure
    fig, ax = plt.subplots(rows, cols)
    # histogram plot
    sns.histplot(data=data_df, x='data', hue='condition',
                 palette=['deeppink', 'dodgerblue'], ax=ax)


    # # for each row
    # for row in range(rows):
    #     # first row plot x1 histogram
    #     if row == 0:
    #         sns.histplot(data=x1, color='deeppink', ax=ax[row])
    #         # ax[row].hist(x1, color='deeppink')
    #     # second row plot x2 histogram
    #     elif row == 1:
    #         sns.histplot(data=x2, color='lightpink', ax=ax[row])
    #         # ax[row].hist(x2, color='lightpink')
    plt.show()
    fig.savefig('/Users/alexanderhsu/Desktop/histogram1_sns.png', dpi=300)


if __name__ == "__main__":
    main()



