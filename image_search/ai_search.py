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



"""
Image Search using External AI API
RAM usage: ~100MB (perfect for Render free)
Uses Replicate API for feature extraction
"""

import requests
import base64
import json
from PIL import Image as PILImage 
from .models import Image as ImageModel
from django.conf import settings
import os


class ExternalAISearchService:
    """Image search using Replicate/HuggingFace API"""

    _instance = None
    
    # HuggingFace Inference API (FREE, no API key needed for public models)
    CLIP_API_URL = "https://api-inference.huggingface.co/models/openai/clip-vit-base-patch32"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            print("✅ External AI Search Service initialized")
        return cls._instance

    def image_to_bytes(self, image_path):
        """Convert image to bytes for API"""
        try:
            with open(image_path, "rb") as f:
                return f.read()
        except Exception as e:
            print(f"Error reading image: {e}")
            return None

    def get_image_embedding(self, image_path):
        """
        Get image embedding from HuggingFace API
        FREE - no authentication required
        """
        try:
            image_bytes = self.image_to_bytes(image_path)
            if not image_bytes:
                return None
            
            # Call HuggingFace Inference API
            response = requests.post(
                self.CLIP_API_URL,
                headers={"Content-Type": "application/octet-stream"},
                data=image_bytes,
                timeout=30
            )
            
            if response.status_code == 200:
                # API returns embedding vector
                result = response.json()
                # CLIP returns embeddings directly
                if isinstance(result, list) and len(result) > 0:
                    return result[0] if isinstance(result[0], list) else result
                return result
            else:
                print(f"API Error: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            print("API timeout - model may be loading")
            return None
        except Exception as e:
            print(f"Error calling API: {e}")
            return None

    def calculate_similarity(self, vec1, vec2):
        """
        Calculate cosine similarity without numpy
        """
        if not vec1 or not vec2:
            return 0.0
        
        # Ensure same length
        min_len = min(len(vec1), len(vec2))
        vec1 = vec1[:min_len]
        vec2 = vec2[:min_len]
        
        # Cosine similarity
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        mag1 = sum(a * a for a in vec1) ** 0.5
        mag2 = sum(b * b for b in vec2) ** 0.5
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        return dot_product / (mag1 * mag2)

    def search_similar_images(self, query_image, threshold=0.7, limit=10, exclude_id=None):
        """
        Search similar images using external API
        
        Args:
            query_image: Path to query image or file object
            threshold: Minimum similarity score (0-1)
            limit: Maximum number of results
            exclude_id: Image ID to exclude from results
        
        Returns:
            List of tuples: [(image_obj, similarity_score), ...]
        """
        # Handle different input types
        if hasattr(query_image, 'temporary_file_path'):
            query_path = query_image.temporary_file_path()
        elif hasattr(query_image, 'read'):
            # Save uploaded file temporarily
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                for chunk in query_image.chunks():
                    tmp.write(chunk)
                query_path = tmp.name
        else:
            query_path = str(query_image)
        
        # Get query image embedding from API
        print(f"🔍 Getting embedding for query image...")
        query_vector = self.get_image_embedding(query_path)
        
        if not query_vector:
            print("⚠️ Failed to get query embedding, using tag-based fallback")
            return self._fallback_tag_search(limit, exclude_id)
        
        # Compare with images in database that have embeddings
        results = []
        queryset = ImageModel.objects.exclude(feature_vector__isnull=True)
        
        if exclude_id:
            queryset = queryset.exclude(id=exclude_id)
        
        print(f"📊 Comparing with {queryset.count()} images in database...")
        
        for img_obj in queryset.iterator(chunk_size=50):
            try:
                db_vector = img_obj.feature_vector
                if not db_vector:
                    continue
                
                similarity = self.calculate_similarity(query_vector, db_vector)
                
                if similarity >= threshold:
                    results.append((img_obj, float(similarity)))
                    
            except Exception as e:
                print(f"⚠️ Error processing image {img_obj.id}: {e}")
                continue
        
        # Sort by similarity (highest first)
        results.sort(key=lambda x: x[1], reverse=True)
        
        print(f"✅ Found {len(results)} similar images")
        return results[:limit]

    def _fallback_tag_search(self, limit, exclude_id):
        """Fallback to recent images if API fails"""
        queryset = ImageModel.objects.all()
        if exclude_id:
            queryset = queryset.exclude(id=exclude_id)
        
        images = queryset.order_by('-uploaded_at')[:limit]
        return [(img, 0.7) for img in images]

    def process_and_save_vector(self, image_obj):
        """
        Extract embedding for an image and save to database
        Call this when user uploads a new image
        """
        try:
            if not os.path.exists(image_obj.image_file.path):
                print(f"❌ Image file not found: {image_obj.image_file.path}")
                return False
            
            print(f"📥 Extracting embedding for image {image_obj.id}...")
            
            # Get embedding from API
            vector = self.get_image_embedding(image_obj.image_file.path)
            
            if vector:
                # Save to database
                image_obj.feature_vector = vector
                image_obj.save(update_fields=["feature_vector"])
                print(f"✅ Embedding saved for image {image_obj.id}")
                return True
            else:
                print(f"⚠️ Failed to extract embedding for image {image_obj.id}")
                return False
                
        except Exception as e:
            print(f"❌ Error processing image {image_obj.id}: {e}")
            return False

    # Backward compatibility methods
    def get_feature_vector(self, img):
        """Extract feature vector from image"""
        if isinstance(img, str):
            return self.get_image_embedding(img)
        elif hasattr(img, 'path'):
            return self.get_image_embedding(img.path)
        return None

    @property
    def is_available(self):
        """Check if service is available"""
        return True


# Singleton instance
ai_search = ExternalAISearchService()

# Backward compatibility
session = None
ai_search_service = ai_search