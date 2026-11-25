# Bộ dữ liệu kiểm thử RAG doanh nghiệp (hư cấu)

Công ty, chính sách, mã định danh và số liệu đều được tạo giả lập. Nạp file này dưới dạng Markdown hoặc dùng file JSONL song song để nạp từng tài liệu.

---

## DOC-001 — Chính sách xử lý yêu cầu hỗ trợ

### 1. Phạm vi và giờ hỗ trợ

Công ty Sao Mai Digital cung cấp nền tảng quản lý đơn hàng và chăm sóc khách hàng cho doanh nghiệp. Bộ phận hỗ trợ tiêu chuẩn làm việc từ thứ Hai đến thứ Sáu, 08:00–18:00 giờ Việt Nam (UTC+7), không bao gồm ngày nghỉ lễ được công ty công bố. Khách hàng thuộc gói Ưu tiên có thể báo sự cố nghiêm trọng qua đường dây trực 24/7. Thời gian phản hồi ban đầu là mục tiêu dịch vụ, không phải cam kết thời gian khắc phục.

### 2. Mức độ ưu tiên

| Mức | Mô tả | Mục tiêu phản hồi ban đầu — gói Ưu tiên |
|---|---|---:|
| P1 — Nghiêm trọng | Dịch vụ chính ngừng hoạt động với phần lớn người dùng và không có giải pháp tạm thời khả thi. | 30 phút |
| P2 — Cao | Chức năng quan trọng bị ảnh hưởng đáng kể nhưng còn dịch vụ hoặc có giải pháp tạm thời hạn chế. | 2 giờ làm việc |
| P3 — Bình thường | Lỗi tính năng không trọng yếu hoặc yêu cầu hướng dẫn cấu hình, sử dụng. | 1 ngày làm việc |
| P4 — Thấp | Câu hỏi chung, lỗi giao diện nhỏ hoặc đề xuất tính năng. | 3 ngày làm việc |

Mức ưu tiên được xác định theo ảnh hưởng nghiệp vụ và khả năng có giải pháp tạm thời, không chỉ theo số lượng người báo lỗi. Khách hàng có thể yêu cầu xem xét lại mức ưu tiên nếu cung cấp quy trình bị ảnh hưởng, số người dùng, thời điểm bắt đầu và giải pháp đã thử.

### 3. Nội dung cần có trong phiếu hỗ trợ

Khi gửi yêu cầu, quản trị viên cần cung cấp:
- Tên tổ chức và mã khách hàng.
- Sản phẩm, môi trường (production hoặc sandbox) và khu vực bị ảnh hưởng.
- Thời điểm bắt đầu, múi giờ và các bước tái hiện lỗi.
- Mã lỗi hoặc correlation ID nếu có.
- Số lượng người dùng bị ảnh hưởng và giải pháp tạm thời đã thử.

Không gửi mật khẩu, mã MFA, số thẻ thanh toán hoặc dữ liệu cá nhân nhạy cảm của khách hàng cuối qua email hay phiếu hỗ trợ thông thường. Nếu đội hỗ trợ cần dữ liệu nhạy cảm, họ sẽ hướng dẫn dùng kênh tải lên bảo mật được phê duyệt.

---

## DOC-002 — Chính sách lưu trữ, xuất và xóa dữ liệu

### 1. Khu vực lưu trữ

Dữ liệu production của Sao Mai Digital được lưu trữ chính tại Singapore. Một số dịch vụ phụ trợ như gửi email, giám sát lỗi và hệ thống hỗ trợ có thể xử lý metadata giới hạn tại khu vực khác. Khách hàng yêu cầu dữ liệu chỉ được lưu trong một quốc gia cần có phụ lục lưu trú dữ liệu được hai bên ký; lựa chọn khu vực trong giao diện không tự tạo cam kết pháp lý.

### 2. Sao lưu và khôi phục

> COPY FROM PDF — Trang 4/18 — SAO MAI DIGITAL — LƯU HÀNH NỘI BỘ

Cơ sở dữ liệu production được sao lưu mã hóa hằng ngày. Bản sao lưu vận hành được giữ tối đa 35 ngày, sau đó tự động hết hạn. Bản sao lưu phục vụ khôi phục dịch vụ, không phải kho lưu trữ dài hạn. RTO/RPO nếu có được quy định theo từng hợp đồng; tài liệu này không đặt ra giá trị chung.

Để xuất dữ liệu, khách hàng gửi phiếu qua Cổng hỗ trợ và nêu khoảng thời gian, loại dữ liệu cùng người nhận được ủy quyền. Định dạng và thời gian bàn giao phụ thuộc khối lượng và phụ lục dịch vụ. Phiếu chỉ hoàn tất khi khách hàng nhận xác nhận bàn giao.

### 3. Kết thúc hợp đồng

Sau khi hợp đồng kết thúc, khách hàng có 45 ngày để yêu cầu xuất dữ liệu, trừ khi hợp đồng quy định khác. Sau thời hạn này, dữ liệu production được xóa trong vòng 30 ngày. Bản sao trong backup hết hạn theo chu kỳ lưu giữ tối đa 35 ngày kể từ ngày tạo bản sao. Nghĩa vụ lưu giữ theo luật hoặc lệnh hợp pháp có thể tạo ngoại lệ.

> COPY FROM PDF — Trang 5/18 — SAO MAI DIGITAL — LƯU HÀNH NỘI BỘ

---

## DOC-003 — Bảng gói dịch vụ và phí sử dụng

### 1. Các gói nền tảng

| Gói | Phí thuê bao tháng | Người dùng bao gồm | Hỗ trợ | Lưu log kiểm toán |
|---|---:|---:|---|---:|
| Cơ bản | 12.000.000 VND | 25 | Tiêu chuẩn | 30 ngày |
| Tăng trưởng | 28.000.000 VND | 100 | Tiêu chuẩn | 90 ngày |
| Doanh nghiệp | Theo báo giá | Theo hợp đồng | Ưu tiên nếu ghi trong đơn hàng | 365 ngày |

Phí chưa bao gồm VAT và phí triển khai. Gói Doanh nghiệp chỉ có hỗ trợ Ưu tiên nếu đơn hàng ký ghi rõ. Log có thể được giữ lâu hơn theo phụ lục riêng.

### 2. Phí vượt mức và thanh toán

Người dùng vượt gói bị tính 180.000 VND/người/tháng, dựa trên số người dùng hoạt động cao nhất trong tháng. Phí vượt mức nằm trên hóa đơn tháng kế tiếp. Hóa đơn thuê bao phát hành đầu mỗi tháng, đến hạn sau 30 ngày trừ khi đơn hàng quy định khác. Khi tranh chấp một phần hóa đơn, khách hàng vẫn thanh toán phần không tranh chấp.

### 3. Gia hạn

Hợp đồng tự gia hạn theo kỳ hạn ghi trên đơn hàng. Khách hàng phải thông báo không gia hạn ít nhất 60 ngày trước ngày gia hạn, trừ khi hợp đồng nêu thời hạn khác. Thay đổi gói chỉ có hiệu lực khi được xác nhận bằng đơn hàng sửa đổi hoặc văn bản hai bên chấp thuận.

---

## DOC-004 — Runbook tích hợp API

### 1. Xác thực và giới hạn

API dùng bearer token riêng cho từng môi trường. Không dùng khóa sandbox trên production. Giới hạn mặc định là 120 yêu cầu mỗi phút cho mỗi tổ chức; một số endpoint có giới hạn thấp hơn trong tài liệu endpoint. Khi vượt giới hạn, API trả HTTP 429 và header `Retry-After` cho biết số giây nên chờ.

### 2. Ví dụ cấu hình webhook

```json
{
  "event": "order.updated",
  "endpoint": "https://merchant.example/webhooks/orders",
  "retries": 5,
  "retry_delay_seconds": 30,
  "signature_header": "X-SM-Signature"
}
```

Webhook được gửi lại tối đa 5 lần, cách nhau 30 giây. Chữ ký HMAC tính trên raw request body; parse rồi serialize lại JSON trước khi xác minh có thể làm chữ ký không khớp. Endpoint cần trả mã 2xx trong 10 giây. Sau lần thử cuối thất bại, sự kiện vào hàng đợi lỗi; quản trị viên yêu cầu gửi lại trong Console.

### 3. Mẫu xử lý retry

```python
import time
import requests

def get_order(url, token):
    for attempt in range(5):
        response = requests.get(
            url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
        )
        if response.status_code != 429:
            response.raise_for_status()
            return response.json()
        wait_seconds = int(response.headers.get("Retry-After", "2"))
        time.sleep(wait_seconds)
    raise RuntimeError("API rate limit persisted after five attempts")
```

Không ghi token vào log. Khi gửi log cho hỗ trợ, che token, email và nội dung đơn hàng. Mã 429 chỉ giới hạn tần suất; mã 401 thường là token thiếu, sai hoặc đã bị thu hồi.

---

## DOC-005 — Quy trình tiếp nhận nhân sự mới

### A. Trước ngày làm việc đầu tiên

Quản lý mở yêu cầu trên cổng Nhân sự ít nhất 3 ngày làm việc trước ngày nhận việc. IT tạo tài khoản sau khi nhận mã nhân viên, phòng ban, quản lý, địa điểm và nhóm quyền được phê duyệt. Tài khoản mặc định không có quyền production.

### B. Trong ngày đầu tiên

1. Nhân viên đăng nhập bằng tài khoản công ty và đăng ký MFA.
2. Quản lý xác nhận thiết bị đã nhận và khóa học an toàn thông tin đã hoàn tất.
3. Chủ hệ thống phê duyệt riêng từng quyền nhạy cảm.
4. IT ghi nhận hoàn tất trong phiếu onboarding.

Nếu thiếu phê duyệt, nhân viên chỉ được cấp quyền cơ bản. Không dùng chung tài khoản để rút ngắn onboarding.

### C. Khi nghỉ việc hoặc chuyển bộ phận

Quản lý gửi yêu cầu trước ngày hiệu lực nếu có thể. Với nghỉ việc đột xuất, quản lý liên hệ IT ngay khi quyết định có hiệu lực. IT vô hiệu hóa đăng nhập công ty trong vòng 4 giờ làm việc kể từ khi nhận yêu cầu đã xác thực; chủ ứng dụng gỡ quyền SaaS trong cùng ngày làm việc. Dữ liệu nghiệp vụ không tự xóa chỉ vì tài khoản bị khóa.

---

## DOC-006 — Trích xuất web: thiết bị và bảo mật

### Thiết bị được quản lý

Thiết bị làm việc với dữ liệu nội bộ phải đăng ký trong hệ thống quản lý thiết bị, bật mã hóa ổ đĩa, khóa màn hình sau tối đa 5 phút không hoạt động và cài bản vá bảo mật quan trọng trong vòng 14 ngày. Thiết bị cá nhân chỉ truy cập email công ty qua ứng dụng được quản lý; không tải dữ liệu khách hàng xuống thiết bị cá nhân.

### Báo cáo sự cố

Người phát hiện email lừa đảo, thiết bị thất lạc hoặc nghi ngờ lộ thông tin phải báo ngay qua nút “Report Security Event” trên Intranet. Nếu thiết bị mất ngoài giờ, gọi số trực nội bộ +84 28 5550 0199. Không tự xóa thiết bị trước khi Security Operations hướng dẫn, trừ khi hệ thống quản lý thiết bị yêu cầu xóa từ xa để ngăn truy cập trái phép.

### Footer trích xuất

Sao Mai Digital | Cổng nhân viên: https://intranet.saomai.example/security | Phiên bản chính sách: 2025-02-01 | Trang 1 / 7

URL và thông tin công ty trong bộ dữ liệu là giả lập; footer có thể bị lặp khi trích xuất PDF/HTML.
