import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import cv2

model = VGG16(weights='imagenet', include_top=False, pooling='avg')

def extract_features(img_path):
    """Извлекает признаки изображения с помощью VGG16"""
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    return features.flatten()

def find_similar_images(query_img_path, folder_path, top_n=5):
    """Находит top_n похожих изображений в папке"""
    query_features = extract_features(query_img_path)

    images = [f for f in os.listdir(folder_path) if f.endswith(('jpg', 'jpeg', 'png'))]

    similarities = []
    for img_name in images:
        img_path = os.path.join(folder_path, img_name)
        try:
            img_features = extract_features(img_path)
            sim = cosine_similarity([query_features], [img_features])[0][0]
            similarities.append((img_name, sim))
        except Exception as e:
            print(f"Ошибка при обработке {img_name}: {e}")

    similarities.sort(key=lambda x: x[1], reverse=True)

    return similarities[:top_n]

def image2array(filelist):
    image_array = []
    for image in filelist[:200]:
        img = io.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224,224))
        image_array.append(img)
    image_array = np.array(image_array)
    image_array = image_array.reshape(image_array.shape[0], 224, 224, 3)
    image_array = image_array.astype('float32')
    image_array /= 255
    return np.array(image_array)

def load_image(path):
    img = image.load_img(path, target_size=model.input_shape[1:3])
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return img, x

folder_path = "photos"  
query_img_path = "picture.jpg"  

similar_images = find_similar_images(query_img_path, folder_path, top_n=100)

print("Самые похожие изображения:")
for img_name, sim in similar_images:
    print(f"{img_name} - сходство: {sim:.2f}")

    img = cv2.imread(os.path.join(folder_path, img_name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.title(f"Сходство: {sim:.2f}")
    plt.axis('off')
    plt.show()


train_data = image2array(urls)
print("Length of training dataset:", train_data.shape)

model = keras.applications.vgg16.VGG16(weights='imagenet', include_top=True)
model.summary()

feat_extractor = Model(inputs=model.input, outputs=model.get_layer("fc2").output)
feat_extractor.summary()

