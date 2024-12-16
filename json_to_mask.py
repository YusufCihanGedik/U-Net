import cv2
import numpy as np
import json
import os

def read_json_labels(label_path):
    with open(label_path, 'r') as file:
        data = json.load(file)
    return data

def process_shapes(data, labels, image_width=1318, image_height=1318):
    class_id_matrix = np.zeros((image_height, image_width), dtype=np.uint8)
    color_id_matrix = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    
    erode_kernel = np.ones((2, 4), np.uint8)
    total_centroid = 0
    # kucuk=99999999999
    kucuk=10
    
    for shapes in data["shapes"]:
        color_id_matrix_2 = np.zeros((image_height, image_width, 3), dtype=np.uint8)

        points_list = shapes["points"]
        label = shapes["label"]
        converted_points = np.array(points_list, np.int32)
        converted_points = converted_points.reshape((-1, 1, 2))
        
        temp_color_id_matrix = np.zeros((image_height, image_width, 3), dtype=np.uint8)
        cv2.fillPoly(temp_color_id_matrix, [converted_points], labels[label][1])
        
        # Belirlenen eşik değeriyle beyaz piksel sayısını kontrol et
        number_of_ones = np.count_nonzero(temp_color_id_matrix[:,:,0] == 255)

        # Eğer beyaz piksel sayısı eşik değerden küçükse erozyon işlemi uygula
        
        if number_of_ones >2000:
            # print(number_of_ones)
            erode_color_id_matrix = cv2.erode(temp_color_id_matrix, erode_kernel, iterations=4)
        else:
            erode_color_id_matrix = temp_color_id_matrix

        color_id_matrix = cv2.add(color_id_matrix, erode_color_id_matrix)
        
        mask_for_class_id = np.zeros((image_height, image_width), dtype=np.uint8)
        cv2.fillPoly(mask_for_class_id, [converted_points], labels[label][0] * 255)
        # print(labels[label][0])
        number_of_ones_class = np.count_nonzero(mask_for_class_id == 255)
        if number_of_ones_class<kucuk:
            kucuk=number_of_ones_class
            
        if number_of_ones_class > 2000:
            erode_mask_for_class_id = cv2.erode(mask_for_class_id, erode_kernel, iterations=4)
        else:
            erode_mask_for_class_id = mask_for_class_id

        class_id_matrix = cv2.add(class_id_matrix, erode_mask_for_class_id)
        if len(np.unique(class_id_matrix)) > 2:
            print("asdasd")
        centroid = np.mean(converted_points, axis=0).astype(int)
        text = f" {number_of_ones_class}"
        cv2.putText(color_id_matrix, text, (centroid[0][0]-20, centroid[0][1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

        cv2.circle(color_id_matrix, (centroid[0][0], centroid[0][1]), 5, (0, 0, 255), -1)
        total_centroid += len(centroid)
    print(f"En kucuk {kucuk}")
    # print(total_centroid)
    return class_id_matrix, color_id_matrix

def display_image_with_mask(image, mask, black_mask):
    alpha = 0.9
    overlay_image = cv2.addWeighted(mask, 1 - alpha, image, alpha, 0)
    resized_mask = cv2.resize(mask, (1000, 1000))
    resized_blackmask = cv2.resize(black_mask, (800, 800))
    resized_image = cv2.resize(image, (800, 800))
    cv2.imshow('Üzerine Yazılmış Görüntü', overlay_image)
    path = r"C:\Users\Gedik\Desktop\gui2\jsontomask"
    cv2.imshow('Mask', resized_mask)
    cv2.imshow('Siyah Maske', resized_blackmask)
    cv2.imshow('Original Image', resized_image)

def display_image(image):
    resized_image = cv2.resize(image, (640, 640))
    cv2.imshow('Original Image', resized_image)

def main_loop(image_folder, label_folder, labels):
    images = sorted([img for img in os.listdir(image_folder) if img.endswith('.png')])
    current_image_index = 0
    total_images = len(images)

    while True:
        image_path = os.path.join(image_folder, images[current_image_index])
        label_path = os.path.join(label_folder, images[current_image_index].replace('.png', '.json'))
        print(image_path)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error loading image {image_path}")
            return

        mask = None
        if os.path.exists(label_path):
            label_data = read_json_labels(label_path)
            black_mask, mask = process_shapes(label_data, labels)

            # Resize mask to match the image size if necessary
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

        if mask is not None:
            display_image_with_mask(image, mask, black_mask)
        else:
            display_image(image)

        key = cv2.waitKey(0) & 0xFF

        if key == ord('d') and current_image_index < total_images - 1:
            current_image_index += 1
        elif key == ord('a') and current_image_index > 0:
            current_image_index -= 1
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_folder = r"C:\Users\Gedik\Desktop\Unet_mask\images"
    label_folder = r"C:\Users\Gedik\Desktop\Unet_mask\mask"
    labels = {"1": [1, (255, 255, 255)], "circle": [2, (151, 255, 255)], "3": [3, (0, 255, 127)]}
    main_loop(image_folder, label_folder, labels)
