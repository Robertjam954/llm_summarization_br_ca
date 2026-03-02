import os
import cv2
import numpy as np
import pytesseract
from PIL import Image
import re
from pathlib import Path

# Set Tesseract command - adjust path as needed for Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s,.!?;:()"\'-]', '', text)
    return text.strip()

def detect_blur(image, threshold=100):
    """Detect if image is blurry using Laplacian variance."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold, laplacian_var

def deblur_image(image):
    """Apply deblurring techniques to image."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Histogram equalization
    gray = cv2.equalizeHist(gray)
    
    # Increase contrast
    gray = cv2.convertScaleAbs(gray, alpha=2.5, beta=50)
    
    # Resize for better readability
    height, width = gray.shape
    scale_factor = 2
    resized = cv2.resize(gray, (width * scale_factor, height * scale_factor), interpolation=cv2.INTER_CUBIC)
    
    # Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(resized, (3, 3), 0)
    
    # Adaptive and Otsu thresholding
    _, binary_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    binary = cv2.bitwise_and(binary_adaptive, binary_otsu)
    
    # Sharpen
    kernel_sharpening = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    binary = cv2.filter2D(binary, -1, kernel_sharpening)
    
    # Morphological operations
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)
    binary = cv2.erode(binary, kernel, iterations=1)
    
    return binary

def extract_text(image):
    """Extract text from image using Tesseract."""
    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image)
    else:
        pil_image = image
    custom_config = r'--oem 3 --psm 6 --dpi 300'
    text = pytesseract.image_to_string(pil_image, config=custom_config)
    return clean_text(text)

def calculate_metrics(extracted_text, ground_truth):
    """Calculate correct, omission, and fabrication metrics."""
    extracted_words = set(extracted_text.lower().split())
    ground_truth_words = set(ground_truth.lower().split())
    
    correct = len(extracted_words & ground_truth_words)
    omissions = len(ground_truth_words - extracted_words)
    fabrications = len(extracted_words - ground_truth_words)
    
    total_gt = len(ground_truth_words) if ground_truth_words else 1
    total_ext = len(extracted_words) if extracted_words else 1
    
    return {
        'correct': correct,
        'omissions': omissions,
        'fabrications': fabrications,
        'precision': correct / total_ext,
        'recall': correct / total_gt
    }

def evaluate_document(image_path, ground_truth_path=None):
    """Evaluate document with and without deblurring."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Load ground truth if available
    ground_truth = ""
    if ground_truth_path and os.path.exists(ground_truth_path):
        with open(ground_truth_path, 'r', encoding='utf-8') as f:
            ground_truth = f.read()
    
    # Check for blur
    is_blurry, blur_score = detect_blur(image)
    
    # Extract text without deblurring
    text_before = extract_text(image)
    
    results = {
        'is_blurry': is_blurry,
        'blur_score': blur_score,
        'text_before_deblur': text_before,
        'metrics_before': calculate_metrics(text_before, ground_truth) if ground_truth else None
    }
    
    # Apply deblurring if blurry
    if is_blurry:
        deblurred = deblur_image(image)
        text_after = extract_text(deblurred)
        results['text_after_deblur'] = text_after
        results['metrics_after'] = calculate_metrics(text_after, ground_truth) if ground_truth else None
    
    return results

def main():
    doc_path = r"C:\Users\jamesr4\OneDrive - Memorial Sloan Kettering Cancer Center\Documents\GitHub\llm_summarization_br_ca\docs\executive_summary.md"
    
    if not os.path.exists(doc_path):
        print(f"File not found: {doc_path}")
        return
    
    # For markdown files, read directly
    if doc_path.endswith('.md'):
        with open(doc_path, 'r', encoding='utf-8') as f:
            content = f.read()
        print("=== Document Content ===")
        print(content)
        print("\n=== Analysis ===")
        print("Markdown file loaded successfully. For image-based OCR evaluation, provide an image file.")
        return
    
    results = evaluate_document(doc_path)
    
    print(f"Blur detected: {results['is_blurry']} (score: {results['blur_score']:.2f})")
    print(f"\n=== Text Before Deblur ===\n{results['text_before_deblur'][:500]}...")
    
    if results.get('metrics_before'):
        m = results['metrics_before']
        print(f"\nMetrics Before: Correct={m['correct']}, Omissions={m['omissions']}, Fabrications={m['fabrications']}")
    
    if results.get('text_after_deblur'):
        print(f"\n=== Text After Deblur ===\n{results['text_after_deblur'][:500]}...")
        if results.get('metrics_after'):
            m = results['metrics_after']
            print(f"\nMetrics After: Correct={m['correct']}, Omissions={m['omissions']}, Fabrications={m['fabrications']}")

if __name__ == "__main__":
    main()
