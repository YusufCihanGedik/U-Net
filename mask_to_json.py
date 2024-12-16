import cv2
import os
import json
import numpy as np
import matplotlib.pyplot as plt

class ImageProcessor:
    def __init__(self, image_path, original_image_name):
        self.image_path = image_path
        self.original_image_name = original_image_name
        self.img = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)  # Maskeyi gri tonlamada yükle
        
        self.process_image()

    def process_image(self):
        # Eşik değeri ile ikili görüntüye çevirme
        _, binary_mask = cv2.threshold(self.img * 255, 1, 255, cv2.THRESH_BINARY)
        print(np.unique(self.img))
        
        # Kontürleri bulma
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Kontürleri çizme
        self.draw_contours(contours)
        
        # Kontürleri görüntüleme
        self.show_contours(contours)

    def draw_contours(self, contours):
        # Çıktı dosya yolu
        output_file = self.generate_output_file_path()
        
        data = {
            "version": "0.3.3",
            "flags": {},
            "shapes": [],
            "imagePath": self.original_image_name,
            "imageData": None,
            "imageHeight": self.img.shape[0],
            "imageWidth": self.img.shape[1],
            "text": ""
        }

        # Kontürleri çiz
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Küçük kontürleri filtreleme
                contour_points = contour.squeeze().tolist()
                shape = {
                    "label": "contour",
                    "points": contour_points,
                    "group_id": None,
                    "shape_type": "polygon",
                    "flags": {}
                }
                data['shapes'].append(shape)

        with open(output_file, 'w') as json_file:
            json.dump(data, json_file, indent=4)

    def show_contours(self, contours):
        # Orijinal görüntüyü renkli olarak yükle (kontürleri çizmek için)
        img_color = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)
        
        # Kontürleri çiz
        cv2.drawContours(img_color, contours, -1, (255, 255, 255), 2)
        
        # Görüntüyü göster
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
        plt.title('Detected Contours')
        plt.axis('off')
        plt.show()

    def generate_output_file_path(self):
        directory = os.path.dirname(self.image_path)
        file_name = os.path.splitext(self.original_image_name)[0] + '.json'
        return os.path.join(directory, file_name)

def process_images_in_folder(folder_path):
    # Belirtilen klasördeki tüm PNG dosyalarını bul
    for file_name in os.listdir(folder_path):
        if file_name.endswith('_mask.png'):
            image_path = os.path.join(folder_path, file_name)
            # Orijinal JPEG dosyasının adını belirle
            original_image_name = file_name.replace('_mask.png', '.jpg')
            processor = ImageProcessor(image_path, original_image_name)
        
# Kullanımı
folder_path = r"C:\Users\Gedik\Desktop\fercam-raw-datasets-20240531T065650Z-001\fercam-raw-datasets\test\yesil_1280q"
process_images_in_folder(folder_path)
