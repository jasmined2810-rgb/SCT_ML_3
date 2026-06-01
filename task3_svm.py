import cv2
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import joblib
import time

print("Starting Task 3: SVM Cats vs Dogs Classification")
start = time.time()

CAT_PATH = 'training_set/cats'
DOG_PATH = 'training_set/dogs'
IMG_SIZE = 64
MAX_IMAGES = 1000

def load_images(folder_path, label, max_images):
    images = []
    count = 0
    for filename in os.listdir(folder_path):
        if count >= max_images:
            break
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                images.append(img.flatten())
                count += 1
    print(f"Loaded {count} images from {folder_path}")
    return images, [label] * count

print("Loading dataset...")
cats, cat_labels = load_images(CAT_PATH, 0, MAX_IMAGES)
dogs, dog_labels = load_images(DOG_PATH, 1, MAX_IMAGES)

X = np.array(cats + dogs)
y = np.array(cat_labels + dog_labels)
print(f"Total: {X.shape[0]} images")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Applying PCA...")
pca = PCA(n_components=150)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print("Training SVM... Wait 1-3 mins")
svm_model = SVC(kernel='rbf', C=10, gamma=0.001)
svm_model.fit(X_train_pca, y_train)

y_pred = svm_model.predict(X_test_pca)
acc = accuracy_score(y_test, y_pred)

print(f"\n=== TASK 3 RESULTS ===")
print(f"Accuracy: {acc*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Cat', 'Dog']))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
plt.imshow(cm, cmap='Blues')
plt.title('SVM: Cats vs Dogs')
plt.colorbar()
plt.xticks([0,1], ['Cat', 'Dog'])
plt.yticks([0,1], ['Cat', 'Dog'])
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=14)
plt.savefig('confusion_matrix.png')
print("Saved: confusion_matrix.png")

joblib.dump(svm_model, 'svm_model.pkl')
joblib.dump(pca, 'pca.pkl')
joblib.dump(scaler, 'scaler.pkl')
print(f"Time taken: {(time.time()-start)/60:.1f} mins")
print("TASK 3 COMPLETE ✅")