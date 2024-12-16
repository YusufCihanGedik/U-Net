import os
import glob
import numpy as np 
import cv2

# Okunacak klasörün yolu
klasor_yolu = r'mask'  # Klasörün yolunu kendi klasör yolunuzla değiştirin

# Klasördeki tüm PNG dosyalarını bul
png_dosyalari = glob.glob(os.path.join(klasor_yolu, '*.png'))

# Her bir PNG dosyasını oku
for dosya_yolu in png_dosyalari:
    with open(dosya_yolu, 'rb') as dosya:
        veri = dosya.read()
        img = cv2.imread(dosya_yolu)
        if  len(np.unique(img))>2:
            print(f"{dosya_yolu} dosyası okundu.")
            print(np.unique(img))
       
       
