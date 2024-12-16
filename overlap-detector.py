import os
import cv2
import json
import numpy as np

# Ana klasör yolunuzu buraya girin.
base_folder = r"E:\Proje repo genel\label_analysis\label_data"
'''
base_folder/
 ├─ cam1/
 │   ├─ image1.png
 │   ├─ image1.json
 │   ├─ image2.png
 │   ├─ image2.json
 │   ... (daha fazla png-json çifti)
 │
 ├─ cam2/
 │   ├─ resimA.png
 │   ├─ resimA.json
 │   ... (başka çiftler)
 │
 └─ output/
     ├─ cam1/
     ├─ cam2/
     ├─ cam3/
     ... (işlenmiş sonuçlar bu klasöre otomatik kaydedilir)

'''
# İşleme sonucu üretilen maskelerin ve ana raporun kaydedileceği klasör.
output_folder = os.path.join(base_folder, "output")
os.makedirs(output_folder, exist_ok=True)

SMALL_OVERLAP_THRESHOLD = 100


def process_image(image_path, json_path, output_dir, folder_name):
    image = cv2.imread(image_path)
    if image is None:
        # Görüntü okunamadıysa terminalde hata ver ve None döndür
        print(f"HATA: Klasör: {folder_name}, Görüntü: {os.path.basename(image_path)} (Görüntü yüklenemedi)")
        return None

    height, width = image.shape[:2]

    # JSON oku
    with open(json_path, 'r') as f:
        data = json.load(f)

    class_id_matrix = np.zeros((height, width), dtype=np.uint8)

    # Poligonları işle
    shapes = data.get("shapes", [])
    for shape in shapes:
        points = shape["points"]
        polygon = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        temp_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(temp_mask, [polygon], 1)
        class_id_matrix += temp_mask

    single_polygon_mask = (class_id_matrix == 1)
    overlap_mask = (class_id_matrix > 1)

    mask = np.zeros((height, width, 3), dtype=np.uint8)
    mask[single_polygon_mask] = (255, 255, 255)
    mask[overlap_mask] = (0, 0, 255)

    overlapping_pixels = np.count_nonzero(overlap_mask)

    # Çakışan alanları bileşenlerine ayır
    overlap_binary = overlap_mask.astype(np.uint8)
    num_components, labels = cv2.connectedComponents(overlap_binary)

    # Küçük çakışma bileşenleri için yeşil daire çiz
    for comp_id in range(1, num_components):
        comp_mask = (labels == comp_id)
        comp_size = np.count_nonzero(comp_mask)
        if 0 < comp_size < SMALL_OVERLAP_THRESHOLD:
            coords = np.column_stack(np.where(comp_mask))
            center_y = int(np.median(coords[:, 0]))
            center_x = int(np.median(coords[:, 1]))
            cv2.circle(mask, (center_x, center_y), 30, (0, 255, 0), 5)

    # Maske kaydet
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    mask_image_path = os.path.join(output_dir, f"{base_name}_mask.png")
    cv2.imwrite(mask_image_path, mask)

    return overlapping_pixels


def main():
    # base_folder içindeki alt klasörleri bul, output hariç
    dirs = [d for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, d)) and d != "output"]

    # Tek bir summary.txt dosyası oluşturalım
    summary_path = os.path.join(output_folder, "summary.txt")
    with open(summary_path, 'w') as summary_file:
        summary_file.write("İşlenen Görüntüler ve Çakışan Piksel Sayıları:\n")
        summary_file.write("Klasör/Görüntü, Çakışan Piksel\n")

        for d in dirs:
            dir_path = os.path.join(base_folder, d)
            # Sadece png dosyalarını alıyoruz
            images = [f for f in os.listdir(dir_path) if f.lower().endswith('.png')]
            out_dir = os.path.join(output_folder, d)
            os.makedirs(out_dir, exist_ok=True)

            for img_name in images:
                json_name = img_name.replace('.png', '.json')
                json_path = os.path.join(dir_path, json_name)
                image_path = os.path.join(dir_path, img_name)

                # JSON yoksa bu .png dosyasını atla
                if not os.path.exists(json_path):
                    continue

                overlapping_pixels = process_image(image_path, json_path, out_dir, d)
                if overlapping_pixels is not None:
                    # Çakışma varsa terminalde uyarı, summary.txt'ye kayıt
                    if overlapping_pixels > 0:
                        print(
                            f"Uyarı: Klasör: {d}, PNG: {img_name}, kesişen alanlar bulundu (Çakışan piksel: {overlapping_pixels})")

                    # Tüm işlenen görüntüler summary.txt'ye yazılır
                    summary_file.write(f"{d}/{img_name}, {overlapping_pixels}\n")


if __name__ == "__main__":
    main()
