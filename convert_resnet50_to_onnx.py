# import tensorflow as tf
# from tensorflow.keras.applications.resnet50 import ResNet50
# import tf2onnx
# import onnx
# import os

# def convert_resnet50_to_onnx():
#     print("\n" + "="*60)
#     print("🚀 CHUYỂN ĐỔI RESNET50: TensorFlow → ONNX")
#     print("="*60)
    
#     # 1. Load TensorFlow model
#     print("\n📥 [1/5] Đang tải ResNet50...")
#     model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
#     print(f"✅ Input shape:  {model.input_shape}")
#     print(f"✅ Output shape: {model.output_shape}")
    
#     # 2. Convert to ONNX
#     print("\n🔧 [2/5] Đang convert sang ONNX...")
#     spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
    
#     model_proto, _ = tf2onnx.convert.from_keras(
#         model, 
#         input_signature=spec, 
#         opset=13
#     )
    
#     # 3. Save ONNX model
#     print("\n💾 [3/5] Đang lưu file...")
#     output_dir = "ml_models"
#     os.makedirs(output_dir, exist_ok=True)
    
#     output_path = os.path.join(output_dir, "resnet50.onnx")
#     onnx.save(model_proto, output_path)
#     print(f"✅ Đã lưu: {output_path}")
    
#     # 4. Verify model
#     print("\n🔍 [4/5] Kiểm tra model...")
#     onnx_model = onnx.load(output_path)
#     onnx.checker.check_model(onnx_model)
#     print("✅ Model hợp lệ!")
    
#     # 5. File info
#     print("\n📊 [5/5] Thông tin:")
#     file_size = os.path.getsize(output_path) / (1024 * 1024)
#     print(f"💾 Kích thước: {file_size:.2f} MB")
#     print(f"📁 Đường dẫn: {os.path.abspath(output_path)}")
    
#     print("\n" + "="*60)
#     print("✅ HOÀN THÀNH!")
#     print("="*60)

# if __name__ == "__main__":
#     try:
#         convert_resnet50_to_onnx()
#     except ImportError:
#         print("\n❌ Thiếu thư viện! Cài đặt:")
#         print("   pip install tensorflow tf2onnx onnx")
#     except Exception as e:
#         print(f"\n❌ LỖI: {str(e)}")

import os
import urllib.request

def download_resnet50_onnx():
    print("\n" + "="*60)
    print("📥 TẢI RESNET50 ONNX")
    print("="*60)
    
    output_dir = "ml_models"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "resnet50.onnx")
    
    if os.path.exists(output_path):
        print("✅ Model đã tồn tại!")
        return
    
    url = "https://huggingface.co/qualcomm/ResNet50/resolve/main/ResNet50.onnx"
    
    print(f"📥 Đang tải từ Hugging Face...")
    urllib.request.urlretrieve(url, output_path)
    
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✅ Hoàn thành! ({file_size:.2f} MB)")

if __name__ == "__main__":
    try:
        download_resnet50_onnx()
    except Exception as e:
        print(f"❌ Lỗi: {e}")