from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import preprocess_input
import numpy as np
import shutil
from sklearn.cluster import KMeans
from collections import Counter
import csv
import glob
import sys

model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
model.summary()

base_path = sys.argv[1]
print(base_path)
# base_path = '/home/aman/Desktop/Mini-Project/KeyFrames'


def get_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)

    resnet50_feature = model.predict(img_data)

    # print(resnet50_feature.shape)
    # print(type(resnet50_feature))
    # print(resnet50_feature.ndim)
    # print(resnet50_feature)

    resnet50_feature = resnet50_feature.tolist()
    resnet50_feature = [j for sub in resnet50_feature for j in sub]
    return resnet50_feature


if __name__ == '__main__':

    allKeyFramesFeat = []

    counter = 0
    dir_path = base_path + '/*.jpg'
    for img_path in glob.iglob(dir_path):

        print(img_path)
        feat = get_features(img_path)
        allKeyFramesFeat.append(feat)
        counter += 1

    print(counter)
    print(len(allKeyFramesFeat))
    # shutil.rmtree("KeyFrames")

    # Clustering
    kmeans = KMeans(n_clusters=5, random_state=0).fit(allKeyFramesFeat)
    print(kmeans.labels_)

    vectorForVideo = []
    for eachCentroid in kmeans.cluster_centers_:
        vectorForVideo.extend(eachCentroid)

    print(vectorForVideo)
    print(len(vectorForVideo))

    # Making feature_vector.csv
    with open("feature_vector.csv", 'a') as outfile:
        writer = csv.writer(outfile, delimiter=' ')
        writer.writerow(vectorForVideo)
