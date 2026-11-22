# Nguồn dữ liệu và giấy phép (Data attribution)

- **Dataset:** Iris.
- **Tác giả gốc:** R. A. Fisher (1936).
- **Kho lưu trữ:** UCI Machine Learning Repository.
- **Trích dẫn:** Fisher, R. (1936). *Iris* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C56C76.
- **Nguồn gốc dữ liệu:** https://archive.ics.uci.edu/dataset/53/iris
- **Tệp nguồn chuẩn:** https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
- **Giấy phép dữ liệu:** Creative Commons Attribution 4.0 International (**CC BY 4.0**): https://creativecommons.org/licenses/by/4.0/

## Lưu ý về cách tái tạo CSV nguồn

Để chạy không cần Internet, dữ liệu `iris_uci_original.csv` được dựng lại bằng
`sklearn.datasets.load_iris` rồi **hoàn nguyên hai bản ghi số 35 và 38**
(index Python 34 và 37) về đúng các giá trị của tệp UCI `iris.data`:

- Bản ghi 35: 4.9, 3.1, 1.5, 0.1, Iris-setosa.
- Bản ghi 38: 4.9, 3.1, 1.5, 0.1, Iris-setosa.

scikit-learn từ phiên bản 0.20 đã sửa hai bản ghi này theo bài báo Fisher;
UCI giữ những giá trị gốc như trên. Phần nhãn không thay đổi. Mã nguồn
`src/prepare_data.py` nêu rõ thao tác này nhằm minh bạch nguồn dữ liệu.

Dữ liệu giữ nguyên 150 quan sát và không làm thay đổi danh tính mẫu. Các
bản ghi trùng giá trị phép đo không bị xóa tùy tiện.

MIT License của mã nguồn **không thay thế** CC BY 4.0 của dữ liệu.
