# Phân tích rủi ro danh mục BTC – Vàng – S&P 500 (2023–2024)

Notebook tiếng Việt trình bày như tiểu luận, gồm tương quan log return, kỳ vọng/
variance/volatility danh mục, đóng góp rủi ro, mô phỏng chuẩn đa biến và bootstrap,
VaR/ES 95%, kiểm tra phân phối và kết luận. Mỗi mục có Markdown giải thích trước
code và nhận xét được cập nhật trực tiếp từ kết quả tính toán.

## Dữ liệu và giả định

- CSV gốc trong `data`: `Bitcoin Historical Data.csv`, `XAU_USD Historical Data.csv`,
  `S&P 500 Historical Data.csv`. Không chỉnh sửa CSV gốc.
- Khoảng thời gian: 01/01/2023–31/12/2024. Dùng `Price` của BTC/USD, XAU/USD và
  S&P 500; chỉ số S&P 500 là đại diện giá cổ phiếu, không gồm cổ tức.
- Lấy giao ngày có giá của cả ba tài sản, không điền giá. Dữ liệu hiện tại cho
  **502 ngày giá** và **501 vector log return**. Return BTC được tính sau đồng bộ
  nên bao gồm biến động cuối tuần giữa hai phiên chung.
- Cơ sở: **20% BTC / 30% Vàng / 50% S&P 500**. So sánh chia đều và 10/40/50.
- Vốn **100.000 USD**, horizon **một phiên chung**, confidence **95%**,
  **100.000 mô phỏng mỗi phương pháp**, seed **42**, quy đổi năm **252 phiên**.
- `r_p ≈ wᵀr` là xấp xỉ log return danh mục theo yêu cầu đề; kỳ vọng năm là
  kỳ vọng log return. Lỗ USD được tính bằng `−V₀ * expm1(r_p)`.
- Bỏ qua phí, thuế, trượt giá; giả định giữ trọng số mục tiêu, độc lập theo thời
  gian và tham số ổn định. Không tối ưu trọng số hoặc dự báo năm 2026.
- Bootstrap lấy nguyên hàng ba return; ES lấy trung bình lỗ chạm/vượt VaR,
  bao gồm ties. P-value Jarque–Bera là xấp xỉ tiệm cận, cần thận trọng với mẫu nhỏ.

## Cài đặt và chạy trên Windows PowerShell

Yêu cầu Python 3.12. Không cần activate môi trường để chạy các lệnh sau:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe script/preprocessing.py
.\.venv\Scripts\python.exe script/run_notebook.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Notebook đã có sẵn tại `notebook_code/portfolio_risk_2023_2024.ipynb`.
`run_notebook.py` khởi tạo kernel sạch bằng đúng Python dùng để gọi lệnh,
thực thi đến cuối và chỉ thay notebook khi toàn bộ cell chạy thành công.
Runtime Jupyter, IPython và Matplotlib được lưu trong dự án, không đăng ký
kernel vào profile người dùng. Kernel cần được phép kết nối socket loopback.

Có thể mở notebook trong VS Code và chọn interpreter `.venv`, hoặc chạy:

```powershell
.\.venv\Scripts\python.exe -m jupyterlab notebook_code/portfolio_risk_2023_2024.ipynb
```

## Đầu ra và các script

| Đầu ra trong `data/processed` | Nội dung |
| --- | --- |
| `prices_2023_2024.csv` | `Date,BTC,GOLD,SP500`, giá đồng bộ, 502 dòng |
| `preprocessing_report.json` | Nguồn, SHA-256 CSV gốc, missing, trùng lặp, ngày bị loại, khoảng cách phiên, cảnh báo |
| `returns_2023_2024.csv` | Cùng bốn cột, log return dạng thập phân, 501 dòng |
| `analysis_results.json` | Kết quả, giả định, phiên bản môi trường và các kiểm tra số học |

Ngày xuất `YYYY-MM-DD`, không làm tròn dữ liệu tính toán. Các bảng notebook
hiển thị đơn vị riêng; số liệu xuất CSV/JSON giữ độ chính xác số thực.
Missing/giá không hữu hạn hoặc không dương, ngày lỗi, ngày trùng khác giá và
dữ liệu sai giai đoạn gây lỗi rõ ràng. Trùng ngày cùng giá được gộp. Biến động
log return trên 10% chỉ được gắn cờ, giữ nguyên để phân tích đuôi phân phối.

- `script/preprocessing.py`: xử lý dữ liệu; mặc định tìm `data` theo vị trí script,
  có tùy chọn `--data-dir` và `--output-dir`.
- `script/run_notebook.py`: thực thi và lưu notebook có output.
- `script/build_notebook.py`: tái tạo nội dung notebook từ nguồn tiếng Việt.
  **Lệnh này xóa output notebook cũ**; chỉ dùng khi cần sửa bản nguồn, rồi chạy lại
  `run_notebook.py`. Nếu chỉnh trực tiếp notebook, không chạy builder để tránh ghi đè.

Sau khi thay CSV gốc, chạy lại notebook để cập nhật cả dữ liệu tiền xử lý,
return, bảng, hình và kết luận. Các kiểm thử snapshot 502/501 dành cho bộ CSV
2023–2024 hiện tại; cần cập nhật kỳ vọng nếu chủ ý thay bộ dữ liệu.

## Kiểm thử

Kiểm thử input bao gồm hai định dạng ngày, BOM, giá phân cách hàng nghìn,
ngoài kỳ, missing, giá sai, ngày trùng và BTC cuối tuần. Kiểm thử kết quả kiểm
tra đủ 27 mục có code, Markdown trước code, toàn bộ cell có output không lỗi,
biểu đồ nhúng, provenance dữ liệu, variance, đóng góp rủi ro, VaR/ES và replay seed.
Notebook kiểm tra mean trong năm sai số chuẩn, SD sai lệch không quá 1%.
VaR chuẩn đa biến còn được đối chiếu với phân vị chuẩn dạng đóng.

## Tham khảo

Nguồn và tài liệu được dẫn trực tiếp tại các mục tương ứng và cuối notebook.
Đề bài nằm trong `context.md`; phạm vi, trọng số và giả định theo kế hoạch
đã thống nhất. Kết quả là bài phân tích mẫu 2023–2024, không phải khuyến nghị đầu tư.
