import subprocess
import os
from glob import glob
import shutil


# Function to make shots
def make_shots(input_video_path):
    os.mkdir("VideoClips")
    subprocess.run(["scenedetect", "-i", input_video_path, "-o", "/home/aman/Desktop/Mini-Project/VideoClips",
                    "detect-content", "-t", "27", "split-video"])


if __name__ == '__main__':

    try:
        os.remove("feature_vector.csv")
        os.remove("label_vector.csv")
    except OSError:
        pass

    # Running algorithm for Accident Videos
    for video_path in glob("/home/aman/Desktop/Mini-Project/Accidents/RoadAccident*.mp4"):

        print(video_path)
        make_shots(video_path)
        os.system("python FrameExtract.py")
        os.system("python KeyFrameExtract.py")
        os.system("python VehicleDistanceDetection.py Accident")
        shutil.rmtree("KeyFrames")
        os.system("python ResNetFeatureExtraction.py Accident")

    # Running algorithm for Non-Accident Videos
    for video_path in glob("/home/aman/Desktop/Mini-Project/NonAccidents/videoplayback*.mp4"):

        print(video_path)
        make_shots(video_path)
        os.system("python FrameExtract.py")
        os.system("python KeyFrameExtract.py")
        os.system("python VehicleDistanceDetection.py NonAccident")
        shutil.rmtree("KeyFrames")
        os.system("python ResNetFeatureExtraction.py NonAccident")
