from flask import Flask, request, jsonify, send_file # type: ignore
from keras.models import load_model
from PIL import Image, ImageChops, ImageEnhance
import numpy as np
import io
import cv2
from matplotlib import pyplot as plt
import base64
import logging

app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.DEBUG)

# Muat model yang telah disimpan
model = load_model('model_casia_run1.h5')
sift = cv2.SIFT_create()

def convert_to_ela_image(image, quality):
    temp_file = io.BytesIO()
    ela_file = io.BytesIO()

    image.save(temp_file, 'JPEG', quality=quality)
    temp_file.seek(0)
    temp_image = Image.open(temp_file)

    ela_image = ImageChops.difference(image, temp_image)

    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff

    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    logging.debug(f"ELA image created with quality {quality}")
    return ela_image

def prepare_image(file_stream):
    image = Image.open(file_stream).convert('RGB')
    logging.debug("Image opened and converted to RGB")
    image_size = (128, 128) # Asumsikan image_size didefinisikan di sini
    ela_image = convert_to_ela_image(image, 91).resize(image_size)
    logging.debug("Image resized and converted to ELA")
    return np.array(ela_image).flatten()/255.0

def compute_ela_cv(image, quality):
    temp_filename = 'temp_file_name.jpg'
    SCALE = 15
    
    # Pastikan image adalah objek PIL.Image
    orig_img = np.array(image)  # Konversi PIL.Image ke NumPy array

    # Mengubah color space dari RGB ke BGR, karena OpenCV menggunakan BGR
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)
    
    # Menyimpan gambar dengan kualitas tertentu
    cv2.imwrite(temp_filename, orig_img, [cv2.IMWRITE_JPEG_QUALITY, quality])

    # Membaca kembali gambar yang dikompresi
    compressed_img = cv2.imread(temp_filename)
    if compressed_img is None:
        logging.error("Failed to load the compressed image properly.")
        return None

    # Menghitung perbedaan antara gambar asli dan gambar yang dikompresi
    diff = SCALE * cv2.absdiff(orig_img, compressed_img)
    logging.debug(f"ELA difference computed with quality {quality}")
    return diff

def ela(image_bytes, quality):
    if quality == 100:
        logging.debug("Quality is 100, returning original image")
        return Image.open(image_bytes)

    # Membaca gambar dari io.BytesIO
    image = Image.open(image_bytes)

    ela_image = compute_ela_cv(image, quality)
    if ela_image is None:
        logging.error("ELA image is None.")
        return None
    logging.debug("ELA image created successfully")
    return Image.fromarray(ela_image.astype('uint8'))

def detect_and_match(image1, image2):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(np.array(image1), cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(np.array(image2), cv2.COLOR_RGB2GRAY)
    logging.debug("Images converted to grayscale")

    # Detect keypoints and compute descriptors
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)
    logging.debug(f"Detected {len(keypoints1)} keypoints in image1 and {len(keypoints2)} keypoints in image2")

    # Feature matching
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    # Apply spatial filter to matches
    filtered_matches = []
    for match in matches:
        pt1 = keypoints1[match.queryIdx].pt
        pt2 = keypoints2[match.trainIdx].pt
        distance = np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
        if distance < 10:  # Threshold for distance
            filtered_matches.append(match)
    logging.debug(f"Filtered matches: {len(filtered_matches)}")

    # Draw all keypoints for image1
    image_with_keypoints = cv2.drawKeypoints(gray1, keypoints1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Draw matches
    matched_image = cv2.drawMatches(gray1, keypoints1, gray2, keypoints2, filtered_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    return matched_image, image_with_keypoints, keypoints1, keypoints2, filtered_matches

def image_to_base64(image):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    encoded_image = base64.b64encode(img_byte_arr).decode('utf-8')
    logging.debug(f"Encoded image to Base64: {encoded_image[:100]}...")  # Log sebagian dari Base64 string untuk verifikasi
    return encoded_image

@app.route('/predict', methods=['POST'])
def predict():
    logging.debug("Request received at /predict")
    logging.debug(request.files)
    logging.debug(request.data)
    if request.method == 'POST':
        if 'image' not in request.files:
            logging.error("No image file in request")
            return jsonify({'error': 'Tidak ada file yang dikirim'})

        file = request.files['image']
        if file.filename == '':
            logging.error("No file selected")
            return jsonify({'error': 'Tidak ada file yang dipilih'})

        if file and file.filename.endswith(('jpg', 'png', 'jpeg')):
            class_names = ['fake', 'real']
            img_bytes = file.read()
            img = io.BytesIO(img_bytes)
            logging.debug("Image file read into BytesIO")
            image = prepare_image(img)
            image = image.reshape(-1, 128, 128, 3)
            predictions = model.predict(image)
            predicted_class = np.argmax(predictions, axis=1)[0]
            logging.debug(f"Prediction made: {class_names[predicted_class]}")
            
            # ELA image
            ela_img_92 = ela(img, 92)
            ela_img_75 = ela(img, 75)

            if ela_img_92 is None or ela_img_75 is None:
                logging.error("Failed to generate ELA images")
                return jsonify({'error': 'Failed to generate ELA images'})

            # Detect and match keypoints
            matched_image, image_with_keypoints, keypoints1, keypoints2, filtered_matches = detect_and_match(ela_img_92, ela_img_75)

            # Convert images to base64
            matched_image_pil = Image.fromarray(matched_image)  # Assuming matched_image is a NumPy array
            keypoints_image_pil = Image.fromarray(image_with_keypoints)

            matched_image_base64 = image_to_base64(matched_image_pil)
            keypoints_image_base64 = image_to_base64(keypoints_image_pil)

            logging.debug(f"Matched image Base64: {matched_image_base64[:100]}...")
            logging.debug(f"Keypoints image Base64: {keypoints_image_base64[:100]}...")

            cv2.imwrite('sift_image1.jpg', matched_image)
            cv2.imwrite('sift_image2.jpg', image_with_keypoints)

            # Membuat respons JSON dengan gambar dan prediksi
            response = {
                'matched_keypoints': matched_image_base64,
                'all_keypoints_92': keypoints_image_base64,
                'prediction': class_names[predicted_class]
            }
            return jsonify(response)
        
        else:
            logging.error("Unsupported file format")
            return jsonify({'error': 'Format file tidak didukung'})

@app.route('/')
def hello_world():
    return 'Selamat datang di server API (deteksi keaslian gambar)'

if __name__ == '__main__':
    app.run(port=4000)