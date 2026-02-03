import os
import shutil
import random

# --- CẤU HÌNH ĐƯỜNG DẪN ---
# Đảm bảo đường dẫn này đúng với cấu trúc bạn đã tạo ở bước trước
ROOT_DIR = './data/isic17'
TRAIN_IMG_DIR = os.path.join(ROOT_DIR, 'train/images')
TRAIN_MASK_DIR = os.path.join(ROOT_DIR, 'train/masks')
VAL_IMG_DIR = os.path.join(ROOT_DIR, 'val/images')
VAL_MASK_DIR = os.path.join(ROOT_DIR, 'val/masks')

# Số lượng tính toán để đạt tỷ lệ 7:3
NUM_TO_MOVE = 495 

def move_files():
    # 1. Kiểm tra số lượng hiện tại
    train_images = [f for f in os.listdir(TRAIN_IMG_DIR) if f.endswith('.jpg') or f.endswith('.png')]
    current_train_count = len(train_images)
    
    print(f"Hiện tại Train có: {current_train_count} ảnh.")
    
    if current_train_count < NUM_TO_MOVE:
        print("Lỗi: Số lượng ảnh trong Train ít hơn số lượng cần chuyển!")
        return

    # 2. Chọn ngẫu nhiên
    print(f"Đang chọn ngẫu nhiên {NUM_TO_MOVE} ảnh để chuyển sang Val...")
    files_to_move = random.sample(train_images, NUM_TO_MOVE)

    count_success = 0
    for img_name in files_to_move:
        # Xác định tên file mask (giả sử cùng tên, chỉ khác đuôi hoặc cùng đuôi png)
        # Lưu ý: Ở bước prepare_data trước, chúng ta đã đổi tên mask trùng tên ảnh (đuôi .png)
        img_id = os.path.splitext(img_name)[0]
        mask_name = img_id + ".png"
        
        src_img = os.path.join(TRAIN_IMG_DIR, img_name)
        dst_img = os.path.join(VAL_IMG_DIR, img_name)
        
        src_mask = os.path.join(TRAIN_MASK_DIR, mask_name)
        dst_mask = os.path.join(VAL_MASK_DIR, mask_name)
        
        # Chỉ chuyển khi tồn tại cả cặp ảnh và mask để đảm bảo tính toàn vẹn
        if os.path.exists(src_img) and os.path.exists(src_mask):
            shutil.move(src_img, dst_img)
            shutil.move(src_mask, dst_mask)
            count_success += 1
        else:
            print(f"Cảnh báo: Không tìm thấy cặp mask cho {img_name}, bỏ qua.")

    print("-" * 30)
    print(f"Đã chuyển thành công: {count_success} cặp ảnh.")
    print(f"Train còn lại: {current_train_count - count_success} (Mục tiêu: ~1505)")
    print(f"Val hiện có (tăng thêm): {len(os.listdir(VAL_IMG_DIR))} (Mục tiêu: ~645)")

if __name__ == "__main__":
    move_files()