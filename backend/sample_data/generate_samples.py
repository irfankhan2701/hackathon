import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json
import os

os.makedirs("backend/sample_data", exist_ok=True)

# Generate synthetic realistic handwritten document image
width, height = 900, 1100
img = Image.new("RGB", (width, height), color=(250, 248, 242))
draw = ImageDraw.Draw(img)

# Draw document header
draw.text((60, 50), "MEDICAL PRESCRIPTION & DIAGNOSIS", fill=(40, 40, 40))
draw.line([(60, 85), (840, 85)], fill=(180, 180, 180), width=2)

# Simulated handwriting text lines
lines = [
    ("Patient: Sarah Connor", (70, 130), False),
    ("Rx: Amoxicillin 500 mg", (70, 220), False),
    ("Take 2 tablets daily for 7 days", (70, 310), False),
    ("Refill: 3 times", (70, 400), True),  # Crossed out candidate
    ("Date: 12/03/2026", (70, 490), False),
    ("BP: 120/80 mmHg", (70, 580), False),
    ("Dr. Alex Rivera", (70, 670), False),
    ("[Margin] Check blood sugar levels", (620, 250), False),
]

for text, (x, y), is_crossed in lines:
    draw.text((x, y), text, fill=(20, 25, 60))
    if is_crossed:
        # Draw crossed-out stroke over ink
        draw.line([(x - 10, y + 12), (x + 280, y + 15)], fill=(30, 30, 40), width=4)
        draw.line([(x - 5, y + 25), (x + 270, y + 5)], fill=(30, 30, 40), width=3)

# Save sample image
img.save("backend/sample_data/sample_handwriting.png")

ground_truth = {
    "filename": "sample_handwriting.png",
    "ground_truth_text": "Patient: Sarah Connor Rx: Amoxicillin 500 mg Take 2 tablets daily for 7 days Refill: 3 times Date: 12/03/2026 BP: 120/80 mmHg Dr. Alex Rivera Check blood sugar levels",
}

with open("backend/sample_data/ground_truth.json", "w") as f:
    json.dump(ground_truth, f, indent=2)

print("Sample handwriting image and ground truth dataset generated successfully!")
