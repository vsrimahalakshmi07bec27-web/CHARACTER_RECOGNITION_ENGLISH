# ============================================================
# TEXT RECOGNITION USING CNN + IMAGE SEGMENTATION (FINAL OPTIMIZED)
# Dataset: CHARS74K (Sample001–Sample062)
# Includes augmentation + early stopping + output saving + graphs
# ============================================================

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# ------------------------------------------------------------
# STEP 1: Paths Setup
# ------------------------------------------------------------
train_dir = r"C:\Users\gvelm\OneDrive\Documents\final_character\dataset\GoodImg\Img"
augmented_dir = r"C:\Users\gvelm\OneDrive\Documents\final_character\dataset\AugmentedImg"
test_dir = r"C:\Users\gvelm\OneDrive\Documents\final_character\test_images"

# Folder to save all results
results_dir = r"C:\Users\gvelm\OneDrive\Documents\final_character\results"
annotated_output_dir = os.path.join(results_dir, "annotated_outputs")
os.makedirs(results_dir, exist_ok=True)
os.makedirs(annotated_output_dir, exist_ok=True)

# ------------------------------------------------------------
# STEP 2: Define label mapping (Sample001 - Sample062)
# ------------------------------------------------------------
label_map = {
    1:'0', 2:'1', 3:'2', 4:'3', 5:'4', 6:'5', 7:'6', 8:'7', 9:'8', 10:'9',
    11:'A',12:'B',13:'C',14:'D',15:'E',16:'F',17:'G',18:'H',19:'I',20:'J',
    21:'K',22:'L',23:'M',24:'N',25:'O',26:'P',27:'Q',28:'R',29:'S',30:'T',
    31:'U',32:'V',33:'W',34:'X',35:'Y',36:'Z',
    37:'a',38:'b',39:'c',40:'d',41:'e',42:'f',43:'g',44:'h',45:'i',46:'j',
    47:'k',48:'l',49:'m',50:'n',51:'o',52:'p',53:'q',54:'r',55:'s',56:'t',
    57:'u',58:'v',59:'w',60:'x',61:'y',62:'z'
}
categories = [label_map[i] for i in range(1, 63)]

# ------------------------------------------------------------
# STEP 3: Load Dataset
# ------------------------------------------------------------
def load_data_from_dir(base_dir):
    data, labels = [], []
    for i in range(1, 63):
        folder = f"Sample{str(i).zfill(3)}"
        folder_path = os.path.join(base_dir, folder)
        print(f"📂 Loading folder {folder} ...")

        if not os.path.isdir(folder_path):
            print(f"⚠️ Missing folder: {folder_path}")
            continue

        for file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, file)
            try:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, (32, 32))
                data.append(img)
                labels.append(i - 1)
            except Exception as e:
                print(f"⚠️ Skipped {img_path}: {e}")
                continue

    data = np.array(data).reshape(-1, 32, 32, 1).astype("float32") / 255.0
    labels = np.array(labels)
    print(f"✅ Loaded {len(data)} images from {base_dir}\n")
    return data, labels

# Load both datasets
print("📦 Loading main dataset...")
data, labels = load_data_from_dir(train_dir)
print("📦 Loading augmented dataset...")
aug_data, aug_labels = load_data_from_dir(augmented_dir)

# Merge both
data = np.concatenate((data, aug_data))
labels = np.concatenate((labels, aug_labels))
print(f"🧩 Final dataset size: {len(data)} images\n")

# ------------------------------------------------------------
# STEP 4: Split Data
# ------------------------------------------------------------
X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)
y_train = to_categorical(y_train, num_classes=len(categories))
y_val = to_categorical(y_val, num_classes=len(categories))

# ------------------------------------------------------------
# STEP 5: CNN Model
# ------------------------------------------------------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(32,32,1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(categories), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

# ------------------------------------------------------------
# STEP 6: Train with Augmentation + EarlyStopping
# ------------------------------------------------------------
datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1,
                             height_shift_range=0.1, zoom_range=0.1)
datagen.fit(X_train)

early_stop = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    validation_data=(X_val, y_val),
    epochs=25,
    callbacks=[early_stop]
)

# ------------------------------------------------------------
# STEP 7: Save Model
# ------------------------------------------------------------
model_path = os.path.join(results_dir, "character_recognition_model.h5")
model.save(model_path)
print(f"💾 Model saved successfully at: {model_path}")

# ------------------------------------------------------------
# STEP 8: Accuracy & Loss Graphs
# ------------------------------------------------------------
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

graph_path = os.path.join(results_dir, "training_graphs.png")
plt.tight_layout()
plt.savefig(graph_path)
plt.show()
print(f"📈 Training graphs saved at: {graph_path}\n")

# ------------------------------------------------------------
# STEP 9: Text Recognition Function
# ------------------------------------------------------------
model = load_model(model_path)

def recognize_text_from_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠️ Could not read {image_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours]
    boxes = sorted(boxes, key=lambda b: b[0])  # left-to-right

    recognized_text = ""
    for (x, y, w, h) in boxes:
        if w*h < 100:
            continue
        char_img = thresh[y:y+h, x:x+w]
        char_img = cv2.resize(char_img, (32, 32))
        char_img = char_img.reshape(1, 32, 32, 1).astype("float32") / 255.0

        prediction = np.argmax(model.predict(char_img, verbose=0))
        label = categories[prediction]
        recognized_text += label

        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    output_path = os.path.join(annotated_output_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, img)
    print(f"🖼️ {os.path.basename(image_path)} → {recognized_text}")
    print(f"✅ Annotated output saved at: {output_path}\n")
    return recognized_text

# ------------------------------------------------------------
# STEP 10: Run on Test Images
# ------------------------------------------------------------
print("\n🔍 Running text recognition on test images...\n")
for file in os.listdir(test_dir):
    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        image_path = os.path.join(test_dir, file)
        recognize_text_from_image(image_path)

print("\n🎯 Text Recognition Completed Successfully!")
