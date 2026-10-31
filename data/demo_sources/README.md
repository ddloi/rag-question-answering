# Nguồn văn bản bổ sung cho demo_v3

Hai trang được lấy từ Wikipedia API (`prop=extracts|revisions`), giữ toàn bộ extract trong JSON, không viết nội dung dựa trên đáp án QA.

| JSON | Trang | Revision |
| --- | --- | ---: |
| knucklebones.json | [Knucklebones](https://en.wikipedia.org/w/index.php?title=Knucklebones&oldid=1377923263) | 1377923263 |
| winters_tale.json | [The Winter's Tale](https://en.wikipedia.org/w/index.php?title=The_Winter%27s_Tale&oldid=1379720458) | 1379720458 |

Tác giả: cộng đồng đóng góp Wikipedia; lịch sử sửa đổi và ghi công nằm trên trang nguồn. Nội dung văn bản theo [điều khoản tái sử dụng Wikipedia](https://en.wikipedia.org/wiki/Wikipedia:Reusing_Wikipedia_content) và giấy phép CC BY-SA áp dụng tại nguồn. Phần dẫn xuất dùng trong corpus: làm sạch khoảng trắng, chia 200 từ/chunk, overlap 40 từ và thêm tiêu đề trang vào đầu mỗi chunk. Giữ ghi công và điều khoản nguồn khi phân phối lại văn bản.

Các trang được chọn **sau khi xem lỗi câu hỏi test**, chỉ để minh họa sửa coverage. Không gọi demo_v3 là held-out benchmark. Checksum JSON và artifact dẫn xuất nằm trong `data/demo_v3/manifest.json`; mã xây corpus ở `scripts/build_demo_corpus.py`.
