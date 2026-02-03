import torch
import sys
import platform
import os

def check_hardware():
    print("="*60)
    print("KIỂM TRA CẤU HÌNH PHẦN CỨNG CHO VM-UNET")
    print("="*60)

    # 1. Thông tin Hệ điều hành & Python
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Python Version: {sys.version.split()[0]}")
    
    # 2. Kiểm tra System RAM (RAM máy tính)
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024**3)
        print(f"System RAM: {ram_gb:.2f} GB")
    except ImportError:
        print("System RAM: (Cài đặt thư viện 'psutil' để xem thông tin này)")

    print("-" * 60)

    # 3. Kiểm tra PyTorch & CUDA
    print(f"PyTorch Version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")

    if cuda_available:
        current_device = torch.cuda.current_device()
        print(f"CUDA Version (PyTorch): {torch.version.cuda}")
        print(f"Số lượng GPU: {torch.cuda.device_count()}")
        
        # Lấy thông tin chi tiết GPU 0 (GPU chính)
        props = torch.cuda.get_device_properties(current_device)
        gpu_name = props.name
        total_memory = props.total_memory / (1024**3) # Convert to GB
        
        print(f"\n[CHI TIẾT GPU ĐANG SỬ DỤNG]")
        print(f"  - Tên GPU: {gpu_name}")
        print(f"  - Tổng VRAM: {total_memory:.2f} GB")
        
        # Kiểm tra khả năng tính toán (Compute Capability)
        cc = props.major * 10 + props.minor
        print(f"  - Compute Capability: {props.major}.{props.minor}")
        
        # Khuyến nghị cấu hình cho VM-UNet
        print("\n" + "="*20 + " KHUYẾN NGHỊ CẤU HÌNH " + "="*20)
        if total_memory < 6:
            print("⚠️  VRAM thấp (< 6GB).")
            print("-> Hãy set 'batch_size' trong config_setting.py xuống 4 hoặc 8.")
        elif total_memory < 10:
            print("✅ VRAM trung bình (6GB - 10GB).")
            print("-> Bạn có thể set 'batch_size' khoảng 16.")
        else:
            print("🚀 VRAM tốt (> 10GB).")
            print("-> Bạn có thể set 'batch_size' là 32 hoặc cao hơn.")
            
    else:
        # Check for MPS (Apple Silicon)
        if torch.backends.mps.is_available():
            print(f"\n✅ MPS Available: True")
            print("  - Device: Apple Silicon (M1/M2/M3)")
            print("  - Bạn có thể train sử dụng GPU của chip M1/M2/M3.")
            
            # Basic RAM check for advice (Apple Unified Memory)
            import psutil
            total_ram = psutil.virtual_memory().total / (1024**3)
            print(f"  - Total Unified Memory: {total_ram:.2f} GB")
            
            if total_ram < 16:
                print("⚠️  RAM < 16GB. Nên set 'batch_size' nhỏ (2 hoặc 4) để tránh tràn bộ nhớ.")
            else:
                 print("🚀 RAM >= 16GB. Có thể set 'batch_size' 8 hoặc 16.")

        else:
            print("\n❌ CẢNH BÁO: Không tìm thấy GPU NVIDIA và không hỗ trợ MPS.")
            print("Việc train VM-UNet trên CPU sẽ cực kỳ chậm hoặc không khả thi.")

    print("="*60)

if __name__ == "__main__":
    check_hardware()