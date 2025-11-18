import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50

model = load_model("clothes_classifier_resnet50.h5")

test_gen = ImageDataGenerator(rescale=1./255)

# img = cv2.imread("test2.png")
# img = cv2.resize(img, (224, 224))
# img = img / 255.0
# img = np.expand_dims(img, axis=0)

# pred = model.predict(img)[0][0]

# if pred > 0.5:
#     print("Good Clothes")
# else:
#     print("Bad Clothes")
    

# Predict on test dataset
test_data = test_gen.flow_from_directory(
    "./dataset/split_data/test",
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary",
    shuffle=False
)

pred = model.predict(test_data)
pred_labels = (pred > 0.5).astype(int)

cm = confusion_matrix(test_data.classes, pred_labels)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

print(classification_report(test_data.classes, pred_labels))