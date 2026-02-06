import os
import shutil
from tqdm import tqdm

# --- CẤU HÌNH ĐƯỜNG DẪN (Bạn sửa lại cho đúng nơi bạn lưu file tải về) ---
# Thư mục chứa dữ liệu gốc đã giải nén
SOURCE_TRAIN_IMAGES = "./Downloads/ISIC-2017_Training_Data" 
SOURCE_TRAIN_MASKS = "./Downloads/ISIC-2017_Training_Part1_GroundTruth"
SOURCE_VAL_IMAGES = "./Downloads/ISIC-2017_Validation_Data"
SOURCE_VAL_MASKS = "./Downloads/ISIC-2017_Validation_Part1_GroundTruth"
SOURCE_TEST_IMAGES = "./Downloads/ISIC-2017_Test_v2_Data"
SOURCE_TEST_MASKS = "./Downloads/ISIC-2017_Test_v2_Part1_GroundTruth"

# Thư mục đích (Project của bạn)
TARGET_ROOT = "./data/isic17"

def process_data(src_img_dir, src_mask_dir, split_name):
    """
    Hàm này di chuyển ảnh và mask vào đúng thư mục, đồng thời đổi tên mask
    để khớp với ID của ảnh.
    """
    dest_img_dir = os.path.join(TARGET_ROOT, split_name, "images")
    dest_mask_dir = os.path.join(TARGET_ROOT, split_name, "masks")
    
    os.makedirs(dest_img_dir, exist_ok=True)
    os.makedirs(dest_mask_dir, exist_ok=True)
    
    # Lấy danh sách file ảnh gốc
    images = [f for f in os.listdir(src_img_dir) if f.endswith('.jpg')]
    
    print(f"Đang xử lý tập {split_name}: {len(images)} ảnh...")
    
    count = 0
    for img_name in tqdm(images):
        image_id = os.path.splitext(img_name)[0] # Lấy ID, ví dụ ISIC_0000000
        
        # 1. Copy ảnh gốc
        src_img_path = os.path.join(src_img_dir, img_name)
        dst_img_path = os.path.join(dest_img_dir, img_name)
        shutil.copy2(src_img_path, dst_img_path)
        
        # 2. Tìm và xử lý mask tương ứng
        # Mask của ISIC thường có dạng: ID_segmentation.png
        mask_name_origin = f"{image_id}_segmentation.png"
        src_mask_path = os.path.join(src_mask_dir, mask_name_origin)
        
        # Kiểm tra xem mask có tồn tại không (quan trọng để tránh lỗi data lệch)
        if os.path.exists(src_mask_path):
            # Đổi tên mask thành giống hệt tên ảnh (nhưng giữ đuôi png)
            # Code dataset.py chỉ cần listdir sorted, nhưng để tên giống nhau cho dễ quản lý
            dst_mask_name = f"{image_id}.png" 
            dst_mask_path = os.path.join(dest_mask_dir, dst_mask_name)
            shutil.copy2(src_mask_path, dst_mask_path)
            count += 1
        else:
            print(f"Cảnh báo: Không tìm thấy mask cho ảnh {img_name}")
            # Nếu thiếu mask, bạn có thể chọn xóa ảnh gốc bên đích để data cân bằng
            os.remove(dst_img_path)
            count -= 1

    print(f"Hoàn thành {split_name}. Tổng cặp dữ liệu hợp lệ: {count}")

if __name__ == "__main__":
    # Xử lý tập Train
    # process_data(SOURCE_TRAIN_IMAGES, SOURCE_TRAIN_MASKS, "train")
    
    # Xử lý tập Val
    # process_data(SOURCE_VAL_IMAGES, SOURCE_VAL_MASKS, "val")
    
    # Xử lý tập Test
    process_data(SOURCE_TEST_IMAGES, SOURCE_TEST_MASKS, "test")
    
    print("\nĐã chuẩn bị dữ liệu xong! Cấu trúc thư mục tại ./data/isic17 đã sẵn sàng.")