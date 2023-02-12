import matplotlib.pyplot as plt
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
    x1, y1 = pose_digit1[:50, 0], pose_digit1[:50, 1]
    x2, y2 = pose_digit2[:50, 0], pose_digit2[:50, 1]
    # fig is entire object to save,
    # ax is axis - you can design multiple axes on the same figure
    fig, ax = plt.subplots(rows, cols)
    # scatter plots dots, plot plots dots that connects
    ax.scatter(x1, y1, s=5, color='deeppink')
    ax.scatter(x2, y2, s=10, color='lightpink')
    plt.show()
    fig.savefig('/Users/alexanderhsu/Desktop/visuals1.png', dpi=300)


if __name__ == "__main__":
    main()



