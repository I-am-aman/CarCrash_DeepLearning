import subprocess
import os
from glob import glob


# Function to make shots
def make_shots(var):
    os.mkdir("VideoClips")
    subprocess.run(["scenedetect", "-i", var, "-o", "/home/aman/Desktop/Mini-Project/VideoClips", "detect-content",
                    "-t", "27", "split-video"])


if __name__ == '__main__':

    i = 0

    # Running algorithm for Accident Videos
    for var in glob("/home/aman/Desktop/Mini-Project/RoadAccidents/RoadAccident*.mp4"):

        print(var)
        make_shots(var)
        os.system("python FrameExtract.py")
        os.system("python KeyFrameExtract.py Accident")

        i += 1
        if i == 5:
            break

    i = 0

    # Running algorithm for Non-Accident Videos
    for var in glob("/home/aman/Desktop/Mini-Project/NonAccidents/videoplayback*.mp4"):

        print(var)
        make_shots(var)
        os.system("python FrameExtract.py")
        os.system("python KeyFrameExtract.py NonAccident")

        i += 1
        if i == 5:
            break
