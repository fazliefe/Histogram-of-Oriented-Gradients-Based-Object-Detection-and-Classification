import cv2
import os

def load_image(image_path, grayscale=True):
    """
    Görseli yükler. Grayscale ya da renkli olarak yükleme seçeneği sunar.
    """
    if grayscale:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(image_path)
    
    if img is None:
        print(f"Resim yüklenemedi: {image_path}")
    return img

def resize_image(image, width, height):
    """
    Görseli belirtilen boyutlara yeniden boyutlandırır.
    """
    return cv2.resize(image, (width, height))

def save_image(image, save_path):
    """
    Görseli kaydeder.
    """
    cv2.imwrite(save_path, image)
