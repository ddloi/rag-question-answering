# Đọc tình huống dev để báo cáo

Nguồn: run `results/research_v1_dev`, 200 QA, cùng dataset/model hashes trong REPORT.md. Đây là **agent_self_review định tính**, chọn ví dụ minh họa; không phải mẫu ngẫu nhiên để tính faithfulness, không thay phiếu người chấm 50 QA. Mỗi case có thể tra lại context/source IDs trong predictions.jsonl.gz và lọc bằng cases.csv.

## Đúng theo nhãn và có bằng chứng trực tiếp

**qb_4336 — “Gentoo, Adelie and Macaroni are all species of which bird?”** Canonical `Penguin`; Dense/Hybrid đều sinh `Penguin`. Context đầu là trang Gentoo penguin, mở đầu xác định đây là một penguin species. Dùng ví dụ này để chỉ luồng question → retrieval → context → output; không suy từ một câu đúng thành hệ thống luôn đúng.

## Đáp án cụ thể hơn nhưng EM vẫn bằng 0

**qb_8485 — “A sterlet is what type of creature?”** Canonical `Fish`; model sinh `Sturgeon`. Context xác định sterlet là một loài sturgeon; chuỗi `Fish` không có trong phần context mà T5 thấy. Proxy gắn `context_selection_or_truncation`, nhưng nhãn này chưa chứng minh model trả lời sai nghĩa hoặc context thật sự thiếu bằng chứng. Đây là trường hợp cần xem quan hệ loại/tính cụ thể và người chấm; không tự thêm alias để nâng điểm.

## Ground truth và output đều cần kiểm tra quan hệ

**qb_6193 — “Tennis player Martina Hingis was born in which European country?”** Canonical nguồn là `Switzerland`; Dense/Hybrid sinh `Czech Republic`. Context phân biệt nơi sinh Košice, thời kỳ sống tại Rožnov và việc chuyển đến Switzerland. Hồ sơ [WTA chính thức](https://www.wtatennis.com/legends/80301/martina-hingis) ghi nơi sinh Košice, Slovakia; nhãn `Switzerland` có dấu hiệu nhầm quốc tịch với nơi sinh. Model cũng nhầm nơi lớn lên với nơi sinh, nên không được biện hộ là model đúng chỉ vì nhãn có lỗi.

Đã ghi đề xuất rà nhãn ở `results/research_v1_label_audit_dev.json`, trạng thái agent_proposed, chưa được con người duyệt. Giữ nguyên canonical/aliases và toàn bộ điểm official trong run hiện tại. Nếu sau này sửa nhãn, tạo overlay/version đã duyệt, báo riêng curated score và ảnh hưởng của nhãn; không thay lịch sử benchmark.

## Retrieval chưa tìm đúng quan hệ cần hỏi

**qb_9451 — “The Goya Gate is located at which European museum?”** Canonical `Prado Museum`; Dense sinh `Madrid`, Hybrid sinh `The De Young Museum`. Các passage Dense thực dùng là tiểu sử/phim về Francisco Goya, không nêu bảo tàng trong quan hệ câu hỏi yêu cầu. Địa danh xuất hiện trong nguồn không đồng nghĩa đó là đáp án. Xem top-5 và context sau truncation để giải thích proxy retrieval miss.

## Hybrid đúng EM nhưng chưa đủ để khẳng định faithful

**qb_7781 — “In our solar system, the sun is composed of mostly which gas?”** Canonical `Hydrogen`; Dense sinh `Helium`, Hybrid sinh `Hydrogen`. Hybrid được chấm đúng. Tuy nhiên context 512 token chỉ nêu khoảng 98% gồm hydrogen và helium cùng nhau; nó không cho tỷ lệ riêng đủ để quyết định khí nào chiếm nhiều hơn. Vì vậy đây là ví dụ EM đúng nhưng mức hỗ trợ từ context cần người chấm xem lại, không dùng để tuyên bố citation/faithfulness đã đúng. Việc đổi thứ hạng giúp output khớp nhãn ở câu này chưa chứng minh một cơ chế cải thiện ngữ nghĩa tổng quát.

## Official và canonical chấm khác nhau

**qb_5139 — “Who replaced Iraqi President Ahmed Hassan al-Bakr in July 1979?”** Canonical `Saddam Hussein`; Dense sinh `Saddam`. Official aliases có `saddam`, nên official EM=1 nhưng canonical EM=0. Context thực dùng nói Saddam đảm nhận presidency. Một chuỗi ngắn hơn có thể vẫn chỉ đúng thực thể trong ngữ cảnh; canonical-only không phải oracle ngữ nghĩa. Hybrid ở cùng câu sinh `Vice President`, được giữ nguyên là một đầu ra khác.

## Hybrid cũng có thể làm giảm chất lượng

**qb_7076 — “Which religion uses a domed structure called a Stupa as a shrine?”** Canonical `Buddhism`; Dense sinh `Buddhism`, Hybrid sinh `Christianity`. Dense nhận đoạn có quan hệ Buddhism–stupa. Hybrid nhận đoạn tổng quan shrine liệt kê nhiều tôn giáo, làm mất quan hệ cụ thể trong hai passage đưa vào T5. Đây là ví dụ regression của thay đổi retrieval/context trên cùng checkpoint, không giấu khi trình bày cải tiến.

## Corpus chưa chứa chuỗi canonical

**qb_4157 — “What is the art of hand-making cricket bats called?”** Canonical `Podshaving`; cả hai sinh `Knocking`. Corpus không tìm thấy chuỗi canonical theo kiểm tra đã định nghĩa; context nói về construction và maintenance/knocking-in. Một thuật ngữ liên quan đến bảo dưỡng chưa trả lời được tên nghề/chế tác. Aliases rất rộng như `cricket bat` có thể làm proxy evidence theo official khác canonical; đây là lý do phải ghi answer policy và kiểm tra quan hệ ngữ nghĩa.

## Cách diễn giải kết quả tổng thể

Kiểm tra thủ công sau test, bằng Hybrid trên research_v1, với hai câu người dùng đã báo trước: Antigonus → `The Tempest` (nguồn thực dùng: Shakespeare in Love); Fivestones/Knucklebones/Snobs → `Scrabble` (nguồn: Ring a Ring o' Roses, Scrabble). Cả hai vẫn không khớp đáp án của bộ cũ và chưa được coi là đã sửa đúng. Đầu ra/context đầy đủ ở `results/research_v1_manual_probes.json`; các câu này không được thêm vào train/dev/test hoặc dùng để thay đổi cấu hình đã khóa. Corpus demo_v3 bổ sung sau quan sát lỗi là một version khác; thành công/thất bại ở đó không thay điểm research_v1.

Trên dev, Hybrid EM official cao hơn Dense 3 điểm phần trăm, nhưng paired bootstrap 95% CI là [-2, +8] điểm phần trăm. Khoảng này có 0; chưa đủ để khẳng định cải thiện chắc chắn. Các ví dụ trên phục vụ giải thích, không phải mẫu để ước lượng tỷ lệ lỗi ngữ nghĩa. Kết quả test, phiếu người chấm và Colab nằm ở các artifact riêng.
