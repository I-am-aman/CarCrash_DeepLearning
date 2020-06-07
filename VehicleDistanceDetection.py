import numpy as np
import six.moves.urllib as urllib
import tarfile
import tensorflow as tf
import glob
import datetime
import cv2
import shutil
import sys
import os

if not os.path.exists('RefinedKeyFrames'):
    os.makedirs('RefinedKeyFrames')
if not os.path.exists('RefinedKeyFrames/Accident'):
    os.makedirs('RefinedKeyFrames/Accident')
if not os.path.exists('RefinedKeyFrames/NonAccident'):
    os.makedirs('RefinedKeyFrames/NonAccident')

# ## Object detection imports
# Here are the imports from the object detection module.
os.chdir('/home/aman/Desktop/Mini-Project/models/research/object_detection')
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

localCounter = 0

video_type = sys.argv[1]
print(video_type)
# video_type = 'Accident'

# # Model preparation
# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

# ## Download Model
opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
    file_name = os.path.basename(file.name)
    if 'frozen_inference_graph.pb' in file_name:
        tar_file.extract(file, os.getcwd())


# ## Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.compat.v2.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`,
# we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that
# returns a dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# ## Helper code
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)


def get_keyframe(img_path):
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:

            screen = cv2.imread(img_path, 1)
            screen = cv2.resize(screen, (800,450))
            image_np = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)

            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            # Each box represents a part of the image where a particular object was detected.
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

            # Visualization of the results of a detection.
            # vis_util.visualize_boxes_and_labels_on_image_array(
            #     image_np,
            #     np.squeeze(boxes),
            #     np.squeeze(classes).astype(np.int32),
            #     np.squeeze(scores),
            #     category_index,
            #     use_normalized_coordinates=True,
            #     line_thickness=8)

            for i, b in enumerate(boxes[0]):
                #                 car                    bus                  truck
                if classes[0][i] == 3 or classes[0][i] == 6 or classes[0][i] == 8:
                    if scores[0][i] >= 0.5:
                        mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                        mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                        apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4), 1)
                        # cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)),
                        #                                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

                        if video_type == 'Accident' and apx_distance <= 0.5:
                            if mid_x > 0.3 and mid_x < 0.7:
                                # cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX,
                                #                           1.0, (0,0,255), 3)
                                basename = "/home/aman/Desktop/Mini-Project/RefinedKeyFrames/Accident/frame"
                                suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
                                filename = "_".join([basename, suffix, str(localCounter), ".jpg"])
                                cv2.imwrite(filename, image_np)
                        elif video_type == 'NonAccident' and apx_distance > 0.5:
                            basename = "/home/aman/Desktop/Mini-Project/RefinedKeyFrames/NonAccident/frame"
                            suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
                            filename = "_".join([basename, suffix, str(localCounter), ".jpg"])
                            cv2.imwrite(filename, image_np)

            # cv2.imshow('window',cv2.resize(image_np,(800,450)))
            # if cv2.waitKey(25) & 0xFF == ord('q'):
            #     cv2.destroyAllWindows()
            #     break


if __name__ == '__main__':

    dir_path = "/home/aman/Desktop/Mini-Project/KeyFrames/*.jpg"
    # dir_path = "/home/aman/Desktop/Mini-Project/accident1.jpg"
    for img_path in glob.iglob(dir_path):
        print(img_path)
        get_keyframe(img_path)
        localCounter += 1
