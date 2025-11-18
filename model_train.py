import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, Model
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import cv2
import numpy as np
from tensorflow.keras.models import load_model

data_dir = "../dataset/split_data"
# train_dir = os.path.join(data_dir, "train")
# val_dir   = os.path.join(data_dir, "val")

train_dir = "./dataset/split_data/train"
val_dir   = "./dataset/split_data/val"

img_size = 224
batch = 16
epochs = 10

# Data loaders
train_gen = ImageDataGenerator(rescale=1./255,
                               rotation_range=20,
                               zoom_range=0.2,
                               horizontal_flip=True)

val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    train_dir, target_size=(img_size, img_size),
    batch_size=batch, class_mode="binary"
)

val_data = val_gen.flow_from_directory(
    val_dir, target_size=(img_size, img_size),
    batch_size=batch, class_mode="binary"
)

# Load ResNet50
base = ResNet50(include_top=False, weights="imagenet",
                input_shape=(img_size, img_size, 3))
base.trainable = False

x = layers.GlobalAveragePooling2D()(base.output)
x = layers.Dropout(0.3)(x)
output = layers.Dense(1, activation="sigmoid")(x)

model = Model(base.input, output)

model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=["accuracy"])

history = model.fit(train_data, validation_data=val_data, epochs=epochs)

model.save("clothes_classifier_resnet50_2.h5")



# Extract history data
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

# Plot Accuracy
plt.figure(figsize=(8, 6))
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()
plt.show()

# Plot Loss
plt.figure(figsize=(8, 6))
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.show()

# Predict on test dataset
# test_data = test_gen.flow_from_directory(
#     "./dataset/split_data/test",
#     target_size=(224, 224),
#     batch_size=32,
#     class_mode="binary",
#     shuffle=False
# )

model = load_model("clothes_classifier_resnet50.h5")

img = cv2.imread("test2.png")
img = cv2.resize(img, (224, 224))
img = img / 255.0
img = np.expand_dims(img, axis=0)

# pred = model.predict(img)[0][0]

# if pred > 0.5:
#     print("Good Clothes")
# else:
#     print("Bad Clothes")
    

pred = model.predict(img)
pred_labels = (pred > 0.5).astype(int)

cm = confusion_matrix(test_data.classes, pred_labels)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

print(classification_report(test_data.classes, pred_labels))

