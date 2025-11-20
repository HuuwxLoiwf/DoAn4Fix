# # image_search/ai_search.py
# import numpy as np
# from PIL import Image
# from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
# from tensorflow.keras.preprocessing import image
# from django.core.files.storage import default_storage
# from .model import Image as ImageModel
# import io

# class AIImageSearchService:
#     """Service để tìm kiếm ảnh tương tự bằng AI"""
    
#     def __init__(self):
#         # Load model ResNet50 một lần
#         self.model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    
#     def get_feature_vector(self, img):
#         """
#         Trích xuất vector đặc trưng từ ảnh
#         Args:
#             img: PIL Image object
#         Returns:
#             numpy array: vector đặc trưng
#         """
#         img = img.resize((224, 224))
#         img_array = image.img_to_array(img)
#         img_array = np.expand_dims(img_array, axis=0)
#         img_array = preprocess_input(img_array)
#         features = self.model.predict(img_array, verbose=0)
#         return features.flatten()
    
#     def calculate_similarity(self, vector1, vector2):
#         """
#         Tính độ tương đồng cosine giữa 2 vectors
#         Args:
#             vector1, vector2: numpy arrays
#         Returns:
#             float: độ tương đồng (0-1)
#         """
#         return np.dot(vector1, vector2) / (
#             np.linalg.norm(vector1) * np.linalg.norm(vector2)
#         )
    
#     def search_similar_images(self, query_image, threshold=0.7, limit=10):
#         """
#         Tìm kiếm ảnh tương tự trong database
#         Args:
#             query_image: PIL Image hoặc file path
#             threshold: ngưỡng độ tương đồng tối thiểu (0-1)
#             limit: số lượng kết quả tối đa
#         Returns:
#             list: danh sách (image_obj, similarity_score)
#         """
#         # Trích xuất vector từ ảnh query
#         if isinstance(query_image, str):
#             query_image = Image.open(query_image).convert('RGB')
#         elif not isinstance(query_image, Image.Image):
#             query_image = Image.open(query_image).convert('RGB')
        
#         query_vector = self.get_feature_vector(query_image)
        
#         # Lấy tất cả ảnh từ database
#         all_images = ImageModel.objects.all()
        
#         results = []
#         for img_obj in all_images:
#             try:
#                 # Mở ảnh từ storage
#                 img_path = img_obj.image_file.path
#                 db_image = Image.open(img_path).convert('RGB')
                
#                 # Trích xuất vector
#                 db_vector = self.get_feature_vector(db_image)
                
#                 # Tính độ tương đồng
#                 similarity = self.calculate_similarity(query_vector, db_vector)
                
#                 # Chỉ thêm nếu vượt ngưỡng
#                 if similarity >= threshold:
#                     results.append((img_obj, float(similarity)))
                    
#             except Exception as e:
#                 print(f"Error processing image {img_obj.id}: {str(e)}")
#                 continue
        
#         # Sắp xếp theo độ tương đồng giảm dần
#         results.sort(key=lambda x: x[1], reverse=True)
        
#         # Giới hạn số lượng kết quả
#         return results[:limit]
    
#     def find_duplicates(self, threshold=0.95):
#         """
#         Tìm ảnh trùng lặp trong database
#         Args:
#             threshold: ngưỡng để coi là trùng lặp (0.95 = 95% giống nhau)
#         Returns:
#             list: các cặp ảnh trùng lặp
#         """
#         all_images = list(ImageModel.objects.all())
#         duplicates = []
        
#         # So sánh từng cặp ảnh
#         for i in range(len(all_images)):
#             for j in range(i + 1, len(all_images)):
#                 try:
#                     img1 = Image.open(all_images[i].image_file.path).convert('RGB')
#                     img2 = Image.open(all_images[j].image_file.path).convert('RGB')
                    
#                     vec1 = self.get_feature_vector(img1)
#                     vec2 = self.get_feature_vector(img2)
                    
#                     similarity = self.calculate_similarity(vec1, vec2)
                    
#                     if similarity >= threshold:
#                         duplicates.append({
#                             'image1': all_images[i],
#                             'image2': all_images[j],
#                             'similarity': float(similarity)
#                         })
                        
#                 except Exception as e:
#                     print(f"Error comparing images: {str(e)}")
#                     continue
        
#         return duplicates

# # Khởi tạo service global (singleton)
# ai_search_service = AIImageSearchService()



# image_search/ai_search.py
import os
import numpy as np
from PIL import Image
import onnxruntime as ort
import urllib.request
import logging

logger = logging.getLogger(__name__)

# KHÔNG DÙNG settings.BASE_DIR ở đây nữa → tránh lỗi ImproperlyConfigured
# Dùng đường dẫn tuyệt đối trong container Render
MODEL_PATH = "/app/ml_models/resnet50.onnx"   # ← Render luôn mount project vào /app

def _download_model():
    url = os.getenv("ONNX_MODEL_URL")
    if not url:
        raise RuntimeError("Thiếu ONNX_MODEL_URL trên Render!")
    if os.path.exists(MODEL_PATH):
        logger.info("Model ONNX đã tồn tại, bỏ qua tải lại.")
        return
    logger.info("Đang tải ResNet50 ONNX từ Google Drive (~102MB)...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    urllib.request.urlretrieve(url, MODEL_PATH)
    logger.info("Tải model ONNX thành công!")

# Tải model ngay khi file được import
_download_model()

# Tạo session ONNX
session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name

# Dùng preprocess của Keras (vẫn được, không cần settings)
from tensorflow.keras.applications.resnet50 import preprocess_input

# BÂY GIỜ MỚI IMPORT Django model → lúc này wsgi.py đã set DJANGO_SETTINGS_MODULE rồi
from .models import Image as ImageModel

class AIImageSearchService:
    def _extract_feature(self, pil_img):
        img = pil_img.resize((224, 224))
        arr = np.array(img).astype('float32')
        arr = np.expand_dims(arr, axis=0)
        arr = preprocess_input(arr)
        features = session.run(None, {input_name: arr})[0]
        return features.flatten()

    def calculate_similarity(self, v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)

    def search_similar_images(self, query_image, threshold=0.7, limit=10):
        if isinstance(query_image, str):
            query_image = Image.open(query_image).convert('RGB')
        elif hasattr(query_image, 'read'):
            query_image = Image.open(query_image).convert('RGB')

        query_vec = self._extract_feature(query_image)
        results = []
        for img_obj in ImageModel.objects.all():
            try:
                with img_obj.image_file.open('rb') as f:
                    db_img = Image.open(f).convert('RGB')
                vec = self._extract_feature(db_img)
                sim = self.calculate_similarity(query_vec, vec)
                if sim >= threshold:
                    results.append((img_obj, float(sim)))
            except Exception as e:
                logger.error(f"Lỗi xử lý ảnh: {e}")
                continue
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

# Singleton
ai_search_service = AIImageSearchService()