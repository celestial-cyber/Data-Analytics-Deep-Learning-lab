I've identified all the **unique questions**. Here's the complete Python code for each:

---

## Part A — Spyder IDE / System Operations

### A1. Display Current Working Directory
```python
import os
print("Current Working Directory:", os.getcwd())
```

---

### A2. List All Files in the Current Folder
```python
import os
files = os.listdir(os.getcwd())
print("Files in current folder:")
for f in files:
    print(" -", f)
```

---

### A3. Display System Information
```python
import platform
print("System:", platform.system())
print("Node Name:", platform.node())
print("Release:", platform.release())
print("Version:", platform.version())
print("Machine:", platform.machine())
print("Processor:", platform.processor())
```

---

### A4. Display Current Date and Time
```python
from datetime import datetime
now = datetime.now()
print("Current Date and Time:", now.strftime("%Y-%m-%d %H:%M:%S"))
```

---

### A5. Create a New Folder
```python
import os
folder_name = "MyNewFolder"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
    print(f"Folder '{folder_name}' created successfully.")
else:
    print(f"Folder '{folder_name}' already exists.")
```

---

### A6. Create and Write to a Text File
```python
with open("sample.txt", "w") as f:
    f.write("Hello, this is EduNova AI!\n")
    f.write("Writing to a file using Python.\n")
print("File created and written successfully.")
```

---

### A7. Read a Text File
```python
with open("sample.txt", "r") as f:
    content = f.read()
print("File Content:\n", content)
```

---

### A8. Rename a File
```python
import os
os.rename("sample.txt", "renamed_sample.txt")
print("File renamed successfully.")
```

---

### A9. Delete a File
```python
import os
if os.path.exists("renamed_sample.txt"):
    os.remove("renamed_sample.txt")
    print("File deleted successfully.")
else:
    print("File not found.")
```

---

### A10. Display Python Version
```python
import sys
print("Python Version:", sys.version)
print("Version Info:", sys.version_info)
```

---

### A11. Print Environment Variables
```python
import os
print("Environment Variables:")
for key, value in os.environ.items():
    print(f"  {key} = {value}")
```

---

### A12. Check Whether a File Exists
```python
import os
filename = "sample.txt"
if os.path.exists(filename):
    print(f"'{filename}' EXISTS.")
else:
    print(f"'{filename}' does NOT exist.")
```

---

## Part B — CNN for Computer Vision

### B1. Basic CNN Model (General)
```python
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt

# Load dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Train
history = model.fit(x_train, y_train, epochs=5,
                    validation_data=(x_test, y_test))

# Plot accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('CNN Model Accuracy')
plt.legend()
plt.show()
```

---

### B2. CNN with Edge Detection + ReLU + MaxPooling + Visualization
*(Used for Bike / Animal / Fruit / Bird images)*

```python
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO

# ── STEP 1: Load a 300x300 image ──
# Replace the URL below with any Bike/Animal/Fruit/Bird image URL
# OR load from local file:  Image.open("bike.jpg")

url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Simple_bicycle.svg/300px-Simple_bicycle.svg.png"
response = requests.get(url)
img = Image.open(BytesIO(response.content)).convert("L")  # Grayscale
img = img.resize((300, 300))
img_array = np.array(img, dtype=np.float32)

# ── STEP 2: Prepare tensor (1, H, W, 1) ──
img_tensor = tf.constant(img_array.reshape(1, 300, 300, 1))

# ── STEP 3: Define Edge Detection Kernel ──
edge_kernel = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
], dtype=np.float32).reshape(3, 3, 1, 1)

kernel_tensor = tf.constant(edge_kernel)

# ── STEP 4: Apply Convolution ──
convolved = tf.nn.conv2d(img_tensor, kernel_tensor,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

# ── STEP 5: Apply ReLU Activation ──
activated = tf.nn.relu(convolved)

# ── STEP 6: Apply Max Pooling ──
pooled = tf.nn.max_pool2d(activated,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')

# ── STEP 7: Visualize All 4 Outputs ──
fig, axes = plt.subplots(1, 4, figsize=(18, 5))

axes[0].imshow(img_array, cmap='gray')
axes[0].set_title("Original Image")
axes[0].axis('off')

axes[1].imshow(convolved.numpy().reshape(300, 300), cmap='gray')
axes[1].set_title("After Convolution\n(Edge Detection)")
axes[1].axis('off')

axes[2].imshow(activated.numpy().reshape(300, 300), cmap='gray')
axes[2].set_title("After ReLU\n(Activation)")
axes[2].axis('off')

axes[3].imshow(pooled.numpy().reshape(150, 150), cmap='gray')
axes[3].set_title("After Max Pooling\n(150x150)")
axes[3].axis('off')

plt.suptitle("CNN Pipeline: Convolution → ReLU → MaxPooling", fontsize=14)
plt.tight_layout()
plt.show()
```

> To use a **different image** (Animal, Fruit, Bird), just change the `url` or load locally:
> ```python
> img = Image.open("bird.jpg").convert("L").resize((300, 300))
> ```

---

## Summary of Unique Questions

| # | Question |
|---|----------|
| A1 | Display Current Working Directory |
| A2 | List All Files in Current Folder |
| A3 | Display System Information |
| A4 | Display Current Date and Time |
| A5 | Create a New Folder |
| A6 | Create and Write to a Text File |
| A7 | Read a Text File |
| A8 | Rename a File |
| A9 | Delete a File |
| A10 | Display Python Version |
| A11 | Print Environment Variables |
| A12 | Check Whether a File Exists |
| B1 | Basic CNN Model for Computer Vision |
| B2 | CNN with Edge Detection + ReLU + MaxPooling + Visualization |

All **14 unique programs** covered! Let me know if you need output screenshots or explanations for any of them.