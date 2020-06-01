from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import preprocess_input
import numpy as np

model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
model.summary()

img_path = '/home/aman/Desktop/Mini-Project/accident1.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img_data = image.img_to_array(img)
img_data = np.expand_dims(img_data, axis=0)
img_data = preprocess_input(img_data)

resnet50_feature = model.predict(img_data)

print(resnet50_feature.shape)
print(type(resnet50_feature))
print(resnet50_feature.ndim)
print(resnet50_feature)

