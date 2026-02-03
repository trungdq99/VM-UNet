import os
from PIL import Image
from tqdm import tqdm
import multiprocessing

# --- CẤU HÌNH ---
SOURCE_ROOT = './data/isic17'      # Thư mục chứa data to (đã chia 7:3)
TARGET_ROOT = './data/isic17_256'  # Thư mục đích để lưu data nhỏ
TARGET_SIZE = (256, 256)

def resize_file(args):
    src_path, dst_path, is_mask = args
    
    try:
        with Image.open(src_path) as img:
            # QUAN TRỌNG: Chọn phương pháp resize
            if is_mask:
                # Mask phải dùng NEAREST để giữ nguyên giá trị pixel (0 hoặc 255)
                img_resized = img.resize(TARGET_SIZE, resample=Image.NEAREST)
            else:
                # Ảnh thường dùng BICUBIC hoặc LANCZOS cho mượt
                img_resized = img.resize(TARGET_SIZE, resample=Image.BICUBIC)
            
            # Lưu file
            img_resized.save(dst_path)
            return True
    except Exception as e:
        print(f"Lỗi khi xử lý {src_path}: {e}")
        return False

def prepare_resize_tasks():
    tasks = []
    # Duyệt qua các thư mục con: train/images, train/masks, val/images, val/masks
    for split in ['train', 'val']:
        for type_dir in ['images', 'masks']:
            src_dir = os.path.join(SOURCE_ROOT, split, type_dir)
            dst_dir = os.path.join(TARGET_ROOT, split, type_dir)
            
            # Tạo thư mục đích
            os.makedirs(dst_dir, exist_ok=True)
            
            # Lấy danh sách file
            if not os.path.exists(src_dir):
                continue
                
            files = [f for f in os.listdir(src_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            is_mask = (type_dir == 'masks')
            
            for f in files:
                tasks.append((
                    os.path.join(src_dir, f),
                    os.path.join(dst_dir, f),
                    is_mask
                ))
    return tasks

if __name__ == "__main__":
    print(f"Đang chuẩn bị resize về {TARGET_SIZE}...")
    tasks = prepare_resize_tasks()
    print(f"Tổng số file cần xử lý: {len(tasks)}")
    
    # Sử dụng đa luồng để resize cho nhanh
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    
    # Chạy và hiển thị thanh tiến trình
    for _ in tqdm(pool.imap_unordered(resize_file, tasks), total=len(tasks)):
        pass
        
    pool.close()
    pool.join()
    
    print("\nHoàn tất! Folder mới nằm tại:", TARGET_ROOT)
    print("Hãy nén folder này lại và upload lên Drive.")