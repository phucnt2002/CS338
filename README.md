<!-- Banner -->
<p align="center">
  <a href="https://www.uit.edu.vn/" title="Trường Đại học Công nghệ Thông tin" style="border: none;">
    <img src="https://i.imgur.com/WmMnSRt.png" alt="Trường Đại học Công nghệ Thông tin | University of Information Technology">
  </a>
</p>
<h1 align="center">Nhận dạng - CS338.N21.KHCL</h1>

# GIỚI THIỆU MÔN HỌC
- **Tên môn học:** Nhận dạng
- **Mã môn học:** CS338
- **Mã lớp:** CS338.N21.KHCL
- **Năm học:** HK2 (2022 - 2023)
- **Giảng viên:** Đỗ Văn Tiến
# GIỚI THIỆU NHÓM
| STT | Họ tên | MSSV | Vai trò | Email | Nhiệm vụ |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | Trần Minh Phúc | 20521782 | Nhóm trưởng | 20521782@gm.uit.edu.vn | Đánh giá mô hình, kết quả thực nghiệm, Deploy model lên web   |
| 2 | Nguyễn Thanh Phúc | 20521769 | Thành viên | 20521769@gm.uit.edu.vn | Xử lý model, Deploy model lên web, train model YOLOv5  |
| 3 | Nguyễn Minh Nhật | 20521708 | Thành viên | 20521708@gm.uit.edu.vn | Giới thiệu đồ án, thu thập data, lý thuyết + train model YOLOv5 |
| 4 | Lê Thế Tuấn | 20522113 | Thành viên | 20522113@gm.uit.edu.vn | Lý thuyết OCR, NCLQ  |
# CHỦ ĐỀ ĐỒ ÁN
- **Tên chủ đề:** Nhận dạng biển số xe bằng YOLOv5 + OCR
- **Giới thiệu chủ đề:** Bài toán nhận dạng biển số xe là một lĩnh vực trong lĩnh vực xử lý ảnh và trí tuệ nhân tạo, nhằm nhận dạng và phân tích các thông tin từ hình ảnh biển số xe ô tô hoặc xe máy.
Mục tiêu của bài toán này là nhận dạng chính xác các ký tự trên biển số xe, từ đó xác định được thông tin như khu vực đăng ký, số hiệu xe, loại xe và các thông tin khác.
# CÀI ĐẶT

A step by step series of examples that tell you how to get a development
environment running

Clone project:

    git clone https://github.com/phucnt2002/CS338
    cd YOLOV5
## Local
    pip install --no-cache-dir -r requirements.txt
    python app.py
## Docker:
Build the Docker image:

    docker build -t myapp .

Run the Docker container:

    docker run -p 5000:5000 myapp

You can then access your application at http://localhost:5000
