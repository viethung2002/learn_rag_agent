*Tóm tắt nội dung:*

* *Vấn đề:*
  Docker trên Windows chạy bên trong *WSL2 (Linux VM)* nên dữ liệu Docker được lưu trong file *VHDX* (ổ đĩa ảo). File này *chỉ phình to theo thời gian* (có thể lên đến ~256GB) nhưng *không tự thu nhỏ lại*, dù bạn đã xóa container/image.

* *Vị trí file chiếm dung lượng lớn:*

  * Docker:

    
    %LOCALAPPDATA%\Docker\wsl\data\ext4.vhdx
    
  * WSL (Ubuntu/Debian…):

    
    %LOCALAPPDATA%\Packages\<LinuxDistro>\LocalState\ext4.vhdx
    

* *Dọn rác Docker (bên trong Linux):*

  * Kiểm tra dung lượng:

    
    docker system df
    
  * Xóa toàn bộ rác (container, image, network, build cache):

    
    docker system prune
    

    (hoặc từng phần: docker container/image/network/volume prune)

  👉 Việc này *chỉ xóa dữ liệu bên trong*, *không làm file VHDX nhỏ lại*.

* *Cách thu nhỏ file VHDX (giải phóng dung lượng thật trên Windows):*

  1. Kiểm tra WSL dùng version 2:

     
     wsl --list --verbose
     
  2. Tắt toàn bộ WSL:

     
     wsl --shutdown
     
  3. Mở *DiskPart* (Run as Administrator):

     
     diskpart
     
  4. Chọn file VHDX (ví dụ Docker):

     
     select vdisk file="C:\Users\LENOVO\AppData\Local\Docker\wsl\disk\docker_data.vhdx"
     
  5. Thu gọn ổ đĩa ảo:

     
     compact vdisk
     
  6. Lặp lại cho file VHDX của WSL nếu cần.

* *Kết quả:*
  File VHDX được *thu nhỏ về đúng dung lượng đang dùng* (ví dụ từ 88GB xuống ~49GB), giải phóng hàng chục GB ổ cứng.

* *Lưu ý:*
  Nên *backup file VHDX* nếu có dữ liệu quan trọng trước khi làm.

👉 Kết luận: Docker + WSL trên Windows có thể âm thầm “ăn” rất nhiều dung lượng, và *chỉ dọn Docker thôi là chưa đủ* — cần *compact VHDX* để lấy lại dung lượng thực sự.
