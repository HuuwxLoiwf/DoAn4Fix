# # ai_search_optimized.py
# import numpy as np
# from PIL import Image as PILImage
# from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
# from tensorflow.keras.preprocessing import image
# from .models import Image as ImageModel

# class OptimizedAISearchService:
#     """Optimized AI Image Search Service with caching"""
    
#     _instance = None
#     _model = None
    
#     def __new__(cls):
#         if cls._instance is None:
#             cls._instance = super().__new__(cls)
#             cls._instance._initialize()
#         return cls._instance
    
#     def _initialize(self):
#         """Load model once - singleton pattern"""
#         if self._model is None:
#             self._model = ResNet50(
#                 weights='imagenet', 
#                 include_top=False, 
#                 pooling='avg'
#             )
    
#     @property
#     def model(self):
#         return self._model
    
#     def get_feature_vector(self, img):
#         """Extract feature vector from image"""
#         if isinstance(img, str):
#             img = PILImage.open(img).convert('RGB')
#         elif not isinstance(img, PILImage.Image):
#             img = PILImage.open(img).convert('RGB')
        
#         img = img.resize((224, 224))
#         img_array = image.img_to_array(img)
#         img_array = np.expand_dims(img_array, axis=0)
#         img_array = preprocess_input(img_array)
        
#         features = self.model.predict(img_array, verbose=0)
#         return features.flatten()
    
#     def calculate_similarity(self, vector1, vector2):
#         """Calculate cosine similarity between two vectors"""
#         v1 = np.array(vector1)
#         v2 = np.array(vector2)
#         return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    
#     def search_similar_images(self, query_image, threshold=0.7, limit=10, exclude_id=None):
#         """
#         Search similar images with cached vectors
#         """
#         # Extract query vector
#         query_vector = self.get_feature_vector(query_image)
        
#         # Get images with cached vectors
#         images_with_vectors = ImageModel.objects.exclude(
#             feature_vector__isnull=True
#         )
        
#         if exclude_id:
#             images_with_vectors = images_with_vectors.exclude(id=exclude_id)
        
#         results = []
        
#         for img_obj in images_with_vectors:
#             try:
#                 # Use cached vector
#                 db_vector = np.array(img_obj.feature_vector)
                
#                 # Calculate similarity
#                 similarity = self.calculate_similarity(query_vector, db_vector)
                
#                 if similarity >= threshold:
#                     results.append((img_obj, float(similarity)))
                    
#             except Exception as e:
#                 print(f"Error processing image {img_obj.id}: {str(e)}")
#                 continue
        
#         # Sort by similarity
#         results.sort(key=lambda x: x[1], reverse=True)
        
#         return results[:limit]
    
#     def process_and_save_vector(self, image_obj):
#         """Process image and save vector to database"""
#         try:
#             # Load image from local file
#             img = PILImage.open(image_obj.image_file.path).convert('RGB')
            
#             # Extract feature vector
#             vector = self.get_feature_vector(img)
            
#             # Save to database
#             image_obj.feature_vector = vector.tolist()
#             image_obj.save(update_fields=['feature_vector'])
            
#             return True
            
#         except Exception as e:
#             print(f"Error processing vector for image {img_obj.id}: {str(e)}")
#             return False

# # Singleton instance
# ai_search = OptimizedAISearchService()


# ai_search_optimized.py
# """
# Optimized AI Image Search Service using ONNX Runtime
# Giữ tính năng giống ai_search_optimized.py cũ nhưng nhẹ hơn, không cần TensorFlow
# """

# import numpy as np
# from PIL import Image as PILImage
# import onnxruntime as ort
# from .models import Image as ImageModel
# import os
# from django.conf import settings

# class OptimizedAISearchService:
#     """Optimized AI Image Search Service with ONNX Runtime"""

#     _instance = None
#     _session = None

#     def __new__(cls):
#         if cls._instance is None:
#             cls._instance = super().__new__(cls)
#             cls._instance._initialize()
#         return cls._instance

#     def _initialize(self):
#         """Load ONNX model một lần"""
#         if self._session is None:
#             model_path = os.path.join(settings.BASE_DIR, "ml_models", "resnet50.onnx")
#             if not os.path.exists(model_path):
#                 raise FileNotFoundError(
#                     f"❌ Không tìm thấy ONNX model tại {model_path}. "
#                     f"Hãy chạy convert_resnet50_to_onnx.py trước."
#                 )
#             print(f"📥 Loading ONNX model from: {model_path}")
#             sess_options = ort.SessionOptions()
#             sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
#             sess_options.intra_op_num_threads = 2
#             sess_options.inter_op_num_threads = 2

#             self._session = ort.InferenceSession(
#                 model_path, 
#                 sess_options=sess_options, 
#                 providers=["CPUExecutionProvider"]
#             )
#             print("✅ ONNX model loaded successfully!")
#             print(f"   Providers: {self._session.get_providers()}")

#     @property
#     def model(self):
#         """Để tương thích với code cũ"""
#         return self._session

#     def get_feature_vector(self, img):
#         """Trích xuất feature vector từ ảnh"""
#         if isinstance(img, str):
#             img = PILImage.open(img).convert("RGB")
#         elif not isinstance(img, PILImage.Image):
#             img = PILImage.open(img).convert("RGB")

#         img = img.resize((224, 224))
#         img_array = np.array(img).astype(np.float32)

#         # ImageNet preprocessing (trừ mean)
#         mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
#         img_array -= mean

#         # HWC → CHW
#         img_array = np.transpose(img_array, (2, 0, 1))
#         img_array = np.expand_dims(img_array, axis=0)  # [1, 3, 224, 224]

#         input_name = self._session.get_inputs()[0].name
#         output_name = self._session.get_outputs()[0].name
#         features = self._session.run([output_name], {input_name: img_array})[0]
#         return features.flatten()

#     @staticmethod
#     def calculate_similarity(vector1, vector2):
#         """Cosine similarity giữa 2 vector"""
#         v1 = np.array(vector1)
#         v2 = np.array(vector2)
#         norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
#         if norm_product == 0:
#             return 0.0
#         return float(np.dot(v1, v2) / norm_product)

#     def search_similar_images(self, query_image, threshold=0.7, limit=10, exclude_id=None):
#         """Tìm kiếm ảnh tương tự dựa trên vector đã cache"""
#         query_vector = self.get_feature_vector(query_image)
#         queryset = ImageModel.objects.exclude(feature_vector__isnull=True)
#         if exclude_id:
#             queryset = queryset.exclude(id=exclude_id)

#         results = []
#         for img_obj in queryset.iterator(chunk_size=100):
#             try:
#                 db_vector = np.array(img_obj.feature_vector)
#                 similarity = self.calculate_similarity(query_vector, db_vector)
#                 if similarity >= threshold:
#                     results.append((img_obj, float(similarity)))
#             except Exception as e:
#                 print(f"⚠️ Error processing image {img_obj.id}: {e}")
#                 continue

#         results.sort(key=lambda x: x[1], reverse=True)
#         return results[:limit]

#     def process_and_save_vector(self, image_obj):
#         """Trích xuất vector từ ảnh và lưu vào database"""
#         try:
#             if not os.path.exists(image_obj.image_file.path):
#                 print(f"❌ File not found: {image_obj.image_file.path}")
#                 return False
#             img = PILImage.open(image_obj.image_file.path).convert("RGB")
#             vector = self.get_feature_vector(img)
#             image_obj.feature_vector = vector.tolist()
#             image_obj.save(update_fields=["feature_vector"])
#             return True
#         except Exception as e:
#             print(f"❌ Error processing vector for image {image_obj.id}: {e}")
#             return False

# # Singleton instance để import và dùng
# ai_search = OptimizedAISearchService()


import numpy as np
from PIL import Image as PILImage
import onnxruntime as ort
from .models import Image as ImageModel
import os
from django.conf import settings


class OptimizedAISearchService:
    """Optimized AI Image Search Service with ONNX Runtime"""

    _instance = None
    _session = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Load ONNX model một lần khi khởi động"""
        if self._initialized:
            return
        
        model_path = os.path.join(settings.BASE_DIR, "ml_models", "resnet50.onnx")
        
        try:
            # Kiểm tra file tồn tại
            if not os.path.exists(model_path):
                print(f"❌ ONNX model not found at: {model_path}")
                print("💡 Please run: python convert_resnet50_to_onnx.py")
                print("⚠️ AI Search will run in fallback mode")
                self._session = None
                self._initialized = True
                return
            
            # Load ONNX model
            print(f"📥 Loading ONNX model from: {model_path}")
            
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = 2
            sess_options.inter_op_num_threads = 2

            self._session = ort.InferenceSession(
                model_path, 
                sess_options=sess_options, 
                providers=["CPUExecutionProvider"]
            )
            
            print("✅ ONNX model loaded successfully!")
            print(f"   Input: {self._session.get_inputs()[0].name}")
            print(f"   Output: {self._session.get_outputs()[0].name}")
            print(f"   Providers: {self._session.get_providers()}")
            
        except Exception as e:
            print(f"❌ Error loading ONNX model: {e}")
            print("⚠️ AI Search will run in fallback mode")
            self._session = None
        
        self._initialized = True

    @property
    def model(self):
        """Để tương thích với code cũ"""
        return self._session

    @property
    def is_available(self):
        """Kiểm tra AI search có sẵn không"""
        return self._session is not None

    def get_feature_vector(self, img):
        """Trích xuất feature vector từ ảnh"""
        # Kiểm tra model có sẵn không
        if not self.is_available:
            print("⚠️ AI model not available, returning random vector")
            return np.random.randn(2048).astype(np.float32)
        
        # Xử lý input image
        if isinstance(img, str):
            img = PILImage.open(img).convert("RGB")
        elif not isinstance(img, PILImage.Image):
            img = PILImage.open(img).convert("RGB")

        # Resize về 224x224 (input size của ResNet50)
        img = img.resize((224, 224))
        img_array = np.array(img).astype(np.float32)

        # ImageNet preprocessing (trừ mean)
        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        img_array -= mean

        # HWC → CHW (Height, Width, Channel → Channel, Height, Width)
        img_array = np.transpose(img_array, (2, 0, 1))
        img_array = np.expand_dims(img_array, axis=0)  # [1, 3, 224, 224]

        # Run inference
        input_name = self._session.get_inputs()[0].name
        output_name = self._session.get_outputs()[0].name
        features = self._session.run([output_name], {input_name: img_array})[0]
        
        return features.flatten()

    @staticmethod
    def calculate_similarity(vector1, vector2):
        """Tính cosine similarity giữa 2 vector"""
        v1 = np.array(vector1)
        v2 = np.array(vector2)
        
        norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm_product == 0:
            return 0.0
        
        return float(np.dot(v1, v2) / norm_product)

    def search_similar_images(self, query_image, threshold=0.7, limit=10, exclude_id=None):
        """Tìm kiếm ảnh tương tự dựa trên vector đã cache"""
        if not self.is_available:
            print("⚠️ AI Search not available")
            return []
        
        # Trích xuất vector từ ảnh query
        query_vector = self.get_feature_vector(query_image)
        
        # Lấy tất cả ảnh có feature vector
        queryset = ImageModel.objects.exclude(feature_vector__isnull=True)
        if exclude_id:
            queryset = queryset.exclude(id=exclude_id)

        results = []
        
        # Tính similarity với từng ảnh trong database
        for img_obj in queryset.iterator(chunk_size=100):
            try:
                db_vector = np.array(img_obj.feature_vector)
                similarity = self.calculate_similarity(query_vector, db_vector)
                
                if similarity >= threshold:
                    results.append((img_obj, float(similarity)))
                    
            except Exception as e:
                print(f"⚠️ Error processing image {img_obj.id}: {e}")
                continue

        # Sort theo độ tương đồng giảm dần
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:limit]

    def process_and_save_vector(self, image_obj):
        """Trích xuất vector từ ảnh và lưu vào database"""
        if not self.is_available:
            print("⚠️ AI model not available, skipping vector extraction")
            return False
        
        try:
            # Kiểm tra file ảnh có tồn tại
            if not os.path.exists(image_obj.image_file.path):
                print(f"❌ Image file not found: {image_obj.image_file.path}")
                return False
            
            # Load và trích xuất vector
            img = PILImage.open(image_obj.image_file.path).convert("RGB")
            vector = self.get_feature_vector(img)
            
            # Lưu vào database
            image_obj.feature_vector = vector.tolist()
            image_obj.save(update_fields=["feature_vector"])
            
            print(f"✅ Vector extracted for image {image_obj.id}")
            return True
            
        except Exception as e:
            print(f"❌ Error processing vector for image {image_obj.id}: {e}")
            return False


# Singleton instance để import và dùng
ai_search = OptimizedAISearchService()
