!pip install ultralytics --quiet
!sudo apt install tesseract-ocr -y
!pip install pytesseract opencv-python matplotlib --quiet

# === IMPORT LIBRARIES ===
import cv2
import pytesseract
from ultralytics import YOLO
import matplotlib.pyplot as plt
from google.colab import files
import numpy as np

# === UPLOAD IMAGE ===
uploaded = files.upload()  # Upload a vehicle image
image_path = list(uploaded.keys())[0]
print(f"Using image: {image_path}")

# === READ IMAGE ===
image = cv2.imread(image_path)
if image is None:
    raise ValueError("Error: Image not found or cannot be read.")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# === LOAD YOLOv8 MODEL ===
model = YOLO("yolov8n.pt")  # small pretrained YOLOv8

# === RUN DETECTION ===
results = model(image_rgb)

# === DRAW BOUNDING BOXES AND CROP PLATES ===
plate_images = []
for result in results:
    boxes = result.boxes.xyxy.cpu().numpy()  # x1,y1,x2,y2
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
        plate_crop = image_rgb[y1:y2, x1:x2]
        plate_images.append(plate_crop)

# === DISPLAY DETECTION RESULTS ===
plt.figure(figsize=(10,6))
plt.imshow(image_rgb)
plt.title("YOLO License Plate Detection")
plt.axis('off')
plt.show()

# === OCR RECOGNITION ===
print("=== OCR Results ===")
for idx, plate in enumerate(plate_images):
    plate_gray = cv2.cvtColor(plate, cv2.COLOR_RGB2GRAY)
    text = pytesseract.image_to_string(plate_gray, config='--psm 7')
    print(f"Plate {idx+1}: {text.strip()}")

    # Show cropped plate
    plt.figure(figsize=(5,2))
    plt.imshow(plate_gray, cmap='gray')
    plt.title(f"Cropped Plate {idx+1}")
    plt.axis('off')
    plt.show()

# === OPTIONAL: LABEL TEXT ON ORIGINAL IMAGE ===
for idx, plate in enumerate(plate_images):
    box = results[0].boxes.xyxy.cpu().numpy()[idx]
    x1, y1, x2, y2 = map(int, box)
    text = pytesseract.image_to_string(cv2.cvtColor(plate, cv2.COLOR_RGB2GRAY), config='--psm 7')
    cv2.putText(image_rgb, text.strip(), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

plt.figure(figsize=(10,6))
plt.imshow(image_rgb)
plt.title("Detected Plates with OCR Text")
plt.axis('off')
plt.show()
