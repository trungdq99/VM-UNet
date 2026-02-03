import torch
from torch.utils.data import DataLoader
import timm
from datasets.dataset import NPY_datasets
from tensorboardX import SummaryWriter
from models.vmunet.vmunet import VMUNet

from engine import *
import os
import sys
import shutil
import argparse

from utils import *
from configs.config_setting import setting_config

import warnings
warnings.filterwarnings("ignore")



def main(config, is_resumed=False):

    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    if is_resumed:
        print('#----------Restoring Backup from Drive----------#')
        # Đường dẫn file backup trên Drive (File mà bạn đã lưu là latest_backup.pth)
        drive_backup_path = '/content/drive/MyDrive/VM-UNet/checkpoints_backup/latest_backup.pth'
        
        # Đường dẫn đích (Là file latest.pth trong thư mục mới tạo)
        target_path = resume_model 
        
        if os.path.exists(drive_backup_path):
            try:
                shutil.copy(drive_backup_path, target_path)
                print(f"✅ Đã copy backup từ Drive vào: {target_path}")
                print("Code sẽ tự động nhận diện file này và resume training.")
            except Exception as e:
                print(f"❌ Lỗi khi copy backup: {e}")
        else:
            print(f"⚠️ Cảnh báo: Không tìm thấy file backup tại {drive_backup_path}")
            print("-> Sẽ bắt đầu train từ đầu (Start from scratch).")

    global logger
    logger = get_logger('train', log_dir)
    global writer
    writer = SummaryWriter(config.work_dir + 'summary')

    log_config_info(config, logger)





    print('#----------GPU init----------#')
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    set_seed(config.seed)
    torch.cuda.empty_cache()





    print('#----------Preparing dataset----------#')
    train_dataset = NPY_datasets(config.data_path, config, train=True)
    train_loader = DataLoader(train_dataset,
                                batch_size=config.batch_size, 
                                shuffle=True,
                                pin_memory=True,
                                num_workers=config.num_workers)
    val_dataset = NPY_datasets(config.data_path, config, train=False)
    val_loader = DataLoader(val_dataset,
                                batch_size=1,
                                shuffle=False,
                                pin_memory=True, 
                                num_workers=config.num_workers,
                                drop_last=True)





    print('#----------Prepareing Model----------#')
    model_cfg = config.model_config
    if config.network == 'vmunet':
        model = VMUNet(
            num_classes=model_cfg['num_classes'],
            input_channels=model_cfg['input_channels'],
            depths=model_cfg['depths'],
            depths_decoder=model_cfg['depths_decoder'],
            drop_path_rate=model_cfg['drop_path_rate'],
            load_ckpt_path=model_cfg['load_ckpt_path'],
        )
        model.load_from()
        
    else: raise Exception('network in not right!')
    model = model.cuda()

    cal_params_flops(model, 256, logger)





    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)





    print('#----------Set other params----------#')
    min_loss = 999
    start_epoch = 1
    min_epoch = 1

    if config.only_test_and_save_figs:
        checkpoint = torch.load(config.best_ckpt_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint)
        config.work_dir = config.img_save_path
        if not os.path.exists(config.work_dir + 'outputs/'):
            os.makedirs(config.work_dir + 'outputs/')
        loss = test_one_epoch(
                val_loader,
                model,
                criterion,
                logger,
                config,
            )
        return




    if os.path.exists(resume_model):
        print('#----------Resume Model and Other params----------#')
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        saved_epoch = checkpoint['epoch']
        start_epoch += saved_epoch
        min_loss, min_epoch, loss = checkpoint['min_loss'], checkpoint['min_epoch'], checkpoint['loss']

        log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch}, min_loss: {min_loss:.4f}, min_epoch: {min_epoch}, loss: {loss:.4f}'
        logger.info(log_info)




    step = 0
    print('#----------Training----------#')
    for epoch in range(start_epoch, config.epochs + 1):

        torch.cuda.empty_cache()

        step = train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            epoch,
            step,
            logger,
            config,
            writer
        )

        loss = val_one_epoch(
                val_loader,
                model,
                criterion,
                epoch,
                logger,
                config
            )

        if loss < min_loss:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
            min_loss = loss
            min_epoch = epoch

            # --- ĐOẠN CODE THÊM MỚI: TỰ ĐỘNG BACKUP SANG DRIVE ---
            try:
                # Định nghĩa đường dẫn lưu trên Drive (Bạn sửa lại tên folder cho đúng ý mình)
                # Lưu ý: '/content/drive/MyDrive/...' là đường dẫn chuẩn khi đã mount drive
                drive_backup_dir = '/content/drive/MyDrive/VM-UNet/checkpoints_backup'
                
                # Tạo thư mục trên Drive nếu chưa có
                if not os.path.exists(drive_backup_dir):
                    os.makedirs(drive_backup_dir)

                # Copy file best.pth từ Colab sang Drive
                # Đổi tên file kèm theo số epoch và loss để dễ theo dõi lịch sử
                backup_name = f'best_epoch_{epoch}_loss_{loss:.4f}.pth'
                shutil.copy(os.path.join(checkpoint_dir, 'best.pth'), 
                            os.path.join(drive_backup_dir, backup_name))
                
                # Copy luôn file latest.pth để resume nếu cần
                shutil.copy(os.path.join(checkpoint_dir, 'latest.pth'), 
                            os.path.join(drive_backup_dir, 'latest_backup.pth'))

                print(f"✅ [BACKUP] Đã lưu model tốt nhất (Epoch {epoch}) sang Google Drive: {backup_name}")
            
            except Exception as e:
                print(f"⚠️ [WARNING] Không thể backup sang Drive: {e}")
            # --- KẾT THÚC ĐOẠN CODE THÊM MỚI ---

        torch.save(
            {
                'epoch': epoch,
                'min_loss': min_loss,
                'min_epoch': min_epoch,
                'loss': loss,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, os.path.join(checkpoint_dir, 'latest.pth')) 

    if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
        print('#----------Testing----------#')
        best_weight = torch.load(config.work_dir + 'checkpoints/best.pth', map_location=torch.device('cpu'))
        model.load_state_dict(best_weight)
        loss = test_one_epoch(
                val_loader,
                model,
                criterion,
                logger,
                config,
            )
        os.rename(
            os.path.join(checkpoint_dir, 'best.pth'),
            os.path.join(checkpoint_dir, f'best-epoch{min_epoch}-loss{min_loss:.4f}.pth')
        )      


if __name__ == '__main__':
    # 1. Khởi tạo ArgumentParser
    parser = argparse.ArgumentParser()
    # Thêm argument --is_resumed (nếu có cờ này thì True, không có thì False)
    parser.add_argument('--is_resumed', action='store_true', help='Resume training from Drive backup')
    
    # 2. Lấy các tham số
    args = parser.parse_args()
    
    # 3. Load config và truyền tham số vào main
    config = setting_config
    main(config, is_resumed=args.is_resumed)