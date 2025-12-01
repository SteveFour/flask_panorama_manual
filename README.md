# Panorama Stitcher

Một ứng dụng ghép ảnh nhẹ, chạy trên nền tảng web, được hỗ trợ bởi Python và Flask mà không dùng OpenCV. 

Công cụ này chấp nhận nhiều ảnh chồng chéo (theo thứ tự ngẫu nhiên), tạo ảnh toàn cảnh liền mạch bằng cách sử dụng tính năng khớp SIFT + Homography để ghép ảnh.

# Thành viên nhóm:
| Mã sinh viên  | Họ tên |
| ------------- | -------------    |
| B22DCCN349    | Trần Minh Hoàng  |
| B22DCCN913    | Lê Anh Vũ        |

# Yêu cầu:
- Ngôn ngữ: Python 3.x
- Backend Framework: Flask
- Thư viện: NumPy, PIL, scipy
- Frontend: HTML5, CSS3

# Hướng dẫn chạy:
Chỉ cần chạy file `app.py`
```
python app.py
```

Web sẽ mở ở cổng 5000: http://127.0.0.1:5000/

# Ảnh
## Giao diện chính
<img width="1217" height="682" alt="image" src="https://github.com/user-attachments/assets/63f68ec9-a462-42ab-960a-209f00d103d3" />

## Kết quả chạy
<img width="1230" height="934" alt="image" src="https://github.com/user-attachments/assets/aa793411-96b3-49ca-97f5-ececc4722002" />

# Tài liệu tham khảo & nguồn

- Dataset: https://github.com/flowerDuo/GES-GSP-Stitching/tree/master/Dataset
- First Principle of Computer Vision: https://www.youtube.com/@firstprinciplesofcomputerv3258
- OpenCV Python SIFT Feature Detection (SIFT Algorithm Explained + Code): https://youtu.be/flFbNka62v8
- Projective Transformation: https://youtu.be/2BIzmFD_pRQ
- Introduction to SIFT (Scale-Invariant Feature Transform): https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html
