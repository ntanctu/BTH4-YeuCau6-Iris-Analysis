# BTH04 - Yêu cầu 6 | AI + Python tự động sinh báo cáo khoa học

**Đề tài:** Phân tích các đặc trưng hình thái hoa Iris và xây dựng mô hình phân loại loài từ dữ liệu thực tế.

Đây là mini-project thực hành gồm **dữ liệu gốc, làm sạch, phân tích thống kê, trực quan hóa, mô hình học máy, báo cáo PDF và dashboard tương tác bằng Streamlit**. Bộ dữ liệu không phải dữ liệu mô phỏng.

## 1. Đối chiếu tiêu chí yêu cầu 6

| Tiêu chí | Nội dung đã thực hiện | Tệp |
|---|---|---|
| Dataset thực tế | UCI Iris: 150 mẫu, 4 đặc trưng, 3 loài | `data/iris_uci_original.csv` |
| Làm sạch | Ép kiểu, xác minh thiếu và hợp lệ, chuẩn hóa nhãn, giữ đúng các quan sát | `src/run_pipeline.py` |
| Trực quan hóa | Boxplot, scatter, correlation, confusion matrix | `assets/*.png` |
| Thống kê / mô hình | Mô tả theo nhóm, Pearson, StandardScaler + LogisticRegression | `src/run_pipeline.py` |
| Báo cáo PDF tự sinh | Giới thiệu, mô tả, kết quả, biểu đồ, kết luận, đề xuất | `reports/bao_cao_khoa_hoc_iris.pdf` |
| Dashboard tương tác | Chọn loài, biến, biểu đồ; xem bảng; tải CSV và PDF | `app.py` |
| Quản lý dự án GitHub | README, MIT LICENSE, requirements, .gitignore, dữ liệu và báo cáo | Toàn bộ thư mục |

## 2. Cấu trúc thư mục

```text
BTH04_Yeu_cau_6/
├── app.py                      # Streamlit dashboard
├── dashboard_offline.html      # Bản dashboard HTML ngoại tuyến bổ sung
├── run_all.py                  # Chạy toàn bộ pipeline
├── requirements.txt
├── README.md
├── LICENSE                     # MIT - mã nguồn
├── DATA_LICENSE.md             # CC BY 4.0 - dữ liệu
├── .gitignore
├── data/
│   ├── iris_uci_original.csv
│   └── iris_cleaned.csv
├── src/
│   ├── prepare_data.py         # Phục dựng CSV Iris gốc từ scikit-learn
│   ├── build_offline_dashboard.py # Sinh lại dashboard HTML
│   └── run_pipeline.py         # Làm sạch, huấn luyện, vẽ, xuất PDF
├── assets/                     # 4 biểu đồ PNG
├── reports/
│   ├── bao_cao_khoa_hoc_iris.pdf
│   └── ket_qua_phan_tich.json
└── tests/
    └── test_project.py
```

## 3. Cách chạy trên macOS (Terminal)

**Bước 1:** Giải nén ZIP, mở Terminal, chuyển đến thư mục `BTH04_Yeu_cau_6`.

**Bước 2:** Tạo môi trường ảo và cài đặt:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

**Bước 3:** Chạy tự động toàn bộ các bước phân tích và sinh PDF:

```bash
python3 run_all.py
```

**Bước 4:** Mở dashboard:

```bash
streamlit run app.py
```

Khi hiện địa chỉ `http://localhost:8501`, mở địa chỉ đó bằng trình duyệt. Trên dashboard có thể lọc loài hoa, chọn kiểu biểu đồ, chọn biến cần xem, tải CSV đang lọc hoặc tải **báo cáo PDF toàn bộ bộ dữ liệu**.

**Xem ngay không cần cài Python:** mở tệp `dashboard_offline.html` bằng trình duyệt. Bản này có bộ lọc, biểu đồ tương tác và nút tải CSV/PDF; dashboard bắt buộc theo bài tập vẫn là Streamlit `app.py`.

**Bước 5 (kiểm tra thử):**

```bash
python3 -m unittest discover -s tests -v
```

Nếu cài đặt lỗi trên Python quá mới/cũ, nên dùng Python 3.11 hoặc 3.12. Script sinh PDF dùng font DejaVu Sans đi kèm matplotlib để hiển thị tiếng Việt trên macOS và Linux.

## 4. Phương pháp và độ tin cậy

- **Nguồn:** Fisher, R. (1936). *Iris* [Dataset]. UCI Machine Learning Repository. DOI: https://doi.org/10.24432/C56C76.
- **Biến đầu vào:** Chiều dài/rộng đài hoa và chiều dài/rộng cánh hoa, đơn vị cm.
- **Nhãn:** Setosa, Versicolor, Virginica.
- **Chất lượng:** Dữ liệu được kiểm tra thiếu, sai kiểu, giá trị không hợp lệ và lặp toàn hàng. Trùng phép đo **không nhất thiết** là trùng mẫu, nên được giữ.
- **Mô hình:** Pipeline `StandardScaler` + `LogisticRegression`, chia 70% train / 30% test theo nhãn (`stratify`), `random_state=42`.
- **Đánh giá:** Test Accuracy, Macro F1, ma trận nhầm lẫn; 5-fold CV chỉ trên tập huấn luyện.
- **Giới hạn:** Iris là bộ dữ liệu nhỏ, ba lớp cân bằng. Kết quả không bảo đảm khả năng ứng dụng cho mọi loài cây hay tập dữ liệu mới.

> **Minh bạch phiên bản:** Dữ liệu `load_iris` trong scikit-learn hiện đã sửa hai điểm khác với `iris.data` gốc của UCI. Script `prepare_data.py` khôi phục hai điểm này thành bản gốc UCI. Chi tiết và giấy phép ở `DATA_LICENSE.md`.

## 5. Đưa lên GitHub

Tài khoản GitHub phải có **repository trống mới được tạo** (tên gợi ý `bth04-yeu-cau-6-iris`). Sau khi tạo repository, chạy:

```bash
git init
git add .
git commit -m "BTH04 YC6: automated Iris scientific report and dashboard"
git branch -M main
git remote add origin https://github.com/<TEN-TAI-KHOAN>/bth04-yeu-cau-6-iris.git
git push -u origin main
```

Thay `<TEN-TAI-KHOAN>` bằng tên GitHub của bạn; khi tạo repository **không đánh dấu** tùy chọn tự tạo README/License vì dự án đã có sẵn. Cần xác thực khi push (Git Credential Manager hoặc GitHub CLI). Mã nguồn này được chuẩn bị sẵn cho GitHub; file ZIP **không có nghĩa là đã được đăng lên tài khoản GitHub**.

## 6. Giấy phép

- **Mã nguồn và tài liệu do dự án tạo:** MIT (`LICENSE`).
- **Bộ dữ liệu Iris:** CC BY 4.0, phải ghi nguồn và tác giả (`DATA_LICENSE.md`).

## 7. Kết quả nộp

- Báo cáo khoa học PDF: `reports/bao_cao_khoa_hoc_iris.pdf`.
- Dashboard chạy bằng Streamlit: `app.py`; bản HTML ngoại tuyến xem ngay: `dashboard_offline.html`.
- CSV thật và CSV được chuẩn hóa: `data/`.
- Biểu đồ: `assets/`.
- Code + requirements + giấy phép + README, đủ để tạo repository GitHub.
