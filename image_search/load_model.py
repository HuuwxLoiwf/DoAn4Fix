# image_search/load_model.py
import os
import urllib.request
import logging
from django.conf import settings

logger = logging.getLogger(__name__)

# Đường dẫn chính xác để lưu file ONNX trên Render
MODEL_PATH = os.path.join(settings.BASE_DIR, 'ml_models', 'resnet50.onnx')
MODEL_URL = os.getenv('ONNX_MODEL_URL')

def download_model():
    if not MODEL_URL:
        logger.error("Lỗi: Chưa set ONNX_MODEL_URL trên Render!")
        return None

    if os.path.exists(MODEL_PATH):
        logger.info("Model ONNX đã có sẵn, không cần tải lại.")
        return MODEL_PATH

    logger.info("Đang tải resnet50.onnx từ Google Drive...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        logger.info("Tải model thành công!")
    except Exception as e:
        logger.error(f"Lỗi tải model: {e}")
        return None
    
    return MODEL_PATH

# TỰ ĐỘNG CHẠY KHI SERVER KHỞI ĐỘNG
ONNX_MODEL_PATH = download_model()