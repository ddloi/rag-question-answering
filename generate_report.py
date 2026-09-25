import sys
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
import shutil

sys.stdout.reconfigure(encoding='utf-8')

TEMPLATE_PATH = r"D:\Downloads\MauBaoCao.docx"
OUTPUT_PATH = r"D:\Downloads\Bao_Cao_Tuan_1_De_Tai_20_RAG.docx"
LOCAL_COPY_PATH = r"D:\Projects\Local_Projects\NPL_Test_Project\Bao_Cao_Tuan_1_De_Tai_20_RAG.docx"

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
    tcPr.append(tcMar)

def add_styled_paragraph(doc, text="", style='Normal', space_before=0, space_after=6, line_spacing=1.2, align=WD_ALIGN_PARAGRAPH.JUSTIFY):
    p = doc.add_paragraph(style=style)
    p.alignment = align
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = line_spacing
    if text:
        run = p.add_run(text)
        run.font.name = 'Times New Roman'
        run.font.size = Pt(13)
        run.font.color.rgb = RGBColor(0, 0, 0)
    return p

def add_heading_1(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(14)
    run.font.bold = True
    run.font.color.rgb = RGBColor(0, 51, 102) # Dark navy
    return p

def add_heading_2(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(13)
    run.font.bold = True
    run.font.color.rgb = RGBColor(0, 102, 153)
    return p

def add_heading_3(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(13)
    run.font.bold = True
    run.font.italic = True
    run.font.color.rgb = RGBColor(30, 30, 30)
    return p

def add_bullet(doc, bold_prefix, text):
    p = doc.add_paragraph(style='Normal')
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.2
    p.paragraph_format.left_indent = Inches(0.25)
    
    r_bullet = p.add_run("• ")
    r_bullet.font.name = 'Times New Roman'
    r_bullet.font.size = Pt(13)
    r_bullet.font.bold = True
    
    if bold_prefix:
        r_prefix = p.add_run(bold_prefix + ": ")
        r_prefix.font.name = 'Times New Roman'
        r_prefix.font.size = Pt(13)
        r_prefix.font.bold = True
        
    r_text = p.add_run(text)
    r_text.font.name = 'Times New Roman'
    r_text.font.size = Pt(13)
    return p

def create_report():
    print(f"Đang đọc template từ: {TEMPLATE_PATH}...")
    doc = docx.Document(TEMPLATE_PATH)
    
    # 1. Update Title on Cover page
    for i, p in enumerate(doc.paragraphs):
        if "TÊN ĐỀ TÀI" in p.text:
            p.text = "ĐỀ TÀI 20: XÂY DỰNG VÀ ĐÁNH GIÁ HỆ THỐNG RETRIEVAL-AUGMENTED GENERATION CHO HỎI-ĐÁP"
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in p.runs:
                run.font.name = 'Times New Roman'
                run.font.size = Pt(16)
                run.font.bold = True
                run.font.color.rgb = RGBColor(0, 32, 96)
                
    # 2. Update Table 0 (Phân công công việc Bảng A-1)
    if len(doc.tables) > 0:
        table = doc.tables[0]
        # Data for Week 1 tasks
        tasks = [
            ("1", "Nghiên cứu bài toán (Problem) & Câu hỏi nghiên cứu (Research Questions) trong bài báo NeurIPS 2020", "100% (Chủ trì nghiên cứu & viết báo cáo Chương 1)", "100% (Phối hợp nghiên cứu RQ)", "100% (Phối hợp phân tích Problem)"),
            ("2", "Khảo sát các công trình liên quan (Related Work): REALM, DPR, T5 Closed-book, FiD", "100% (Phối hợp tổng hợp)", "100% (Chủ trì nghiên cứu & viết Chương 2)", "100% (Phối hợp phân tích FiD)"),
            ("3", "Khảo sát các tập dữ liệu (NQ, TriviaQA, Wiki 21M) & Tiền xử lý dữ liệu của bài báo (Preprocessing)", "100% (Phối hợp phân tích chunking)", "100% (Phối hợp phân tích FAISS)", "100% (Chủ trì nghiên cứu & viết Chương 3)"),
            ("4", "Tổng hợp và biên tập báo cáo tiến độ Tuần 1, lập kế hoạch Tuần 2 (25/9 - 3/10)", "100% (Chủ trì tổng hợp & soát lỗi)", "100% (Review kỹ thuật & mô hình)", "100% (Review tài liệu tham khảo)")
        ]
        
        # Ensure table has enough rows
        while len(table.rows) < len(tasks) + 1:
            table.add_row()
            
        for r_idx, task_data in enumerate(tasks):
            row = table.rows[r_idx + 1]
            for c_idx, val in enumerate(task_data):
                cell = row.cells[c_idx]
                cell.text = val
                cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
                set_cell_margins(cell, top=120, bottom=120, left=150, right=150)
                p = cell.paragraphs[0]
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_idx in [0, 2, 3, 4] else WD_ALIGN_PARAGRAPH.LEFT
                for r in p.runs:
                    r.font.name = 'Times New Roman'
                    r.font.size = Pt(11)

    # 3. Add Page Break before Content
    doc.add_page_break()
    
    # --- HEADER TIẾN ĐỘ TUẦN 1 ---
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(12)
    p_title.paragraph_format.space_after = Pt(12)
    r = p_title.add_run("BÁO CÁO TIẾN ĐỘ TUẦN 1\n(19/09/2026 – 25/09/2026)")
    r.font.name = 'Times New Roman'
    r.font.size = Pt(18)
    r.font.bold = True
    r.font.color.rgb = RGBColor(0, 51, 102)

    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_after = Pt(18)
    r_sub = p_sub.add_run("Đề tài 20: Xây dựng và đánh giá hệ thống Retrieval-Augmented Generation cho hỏi-đáp\n"
                          "Bài báo nền tảng: Lewis et al., Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (NeurIPS 2020)\n"
                          "Nội dung trọng tâm: Problem + Research Questions + Related Work + Dataset & Preprocessing")
    r_sub.font.name = 'Times New Roman'
    r_sub.font.size = Pt(12)
    r_sub.font.italic = True

    # =========================================================================
    # CHƯƠNG 1
    # =========================================================================
    add_heading_1(doc, "CHƯƠNG 1: BÀI TOÁN NGHIÊN CỨU VÀ CÂU HỎI NGHIÊN CỨU (PROBLEM & RESEARCH QUESTIONS)")
    
    add_heading_2(doc, "1.1. Bối cảnh và Bài toán đặt ra trong bài báo (Problem Formulation)")
    add_styled_paragraph(doc, 
        "Trong những năm gần đây, sự bùng nổ của các mô hình ngôn ngữ lớn được tiền huấn luyện (Pre-trained Language Models - PLMs) "
        "như BERT, RoBERTa, BART và T5 đã mang lại những bước nhảy vọt mang tính cách mạng trong lĩnh vực Xử lý Ngôn ngữ Tự nhiên (NLP). "
        "Các mô hình này học cách biểu diễn ngữ nghĩa và tích lũy một khối lượng tri thức khổng lồ về thế giới thực thông qua quá trình "
        "học tự giám sát (self-supervised learning) trên hàng trăm gigabyte văn bản mở. Tuy nhiên, toàn bộ lượng tri thức này được lưu trữ "
        "ẩn bên trong hàng trăm triệu hoặc hàng tỷ trọng số của mạng nơ-ron — cơ chế này được bài báo định nghĩa là Bộ nhớ tham số (Parametric Memory)."
    )
    add_styled_paragraph(doc,
        "Bài báo của Patrick Lewis và các cộng sự tại Facebook AI Research (NeurIPS 2020) chỉ ra rằng: Đối với các tác vụ chuyên sâu tri thức "
        "(Knowledge-Intensive NLP Tasks) — các bài toán mà con người cũng không thể giải quyết thỏa đáng nếu không tra cứu tài liệu dẫn chứng, "
        "tiêu biểu như Hỏi - Đáp trong miền mở (Open-Domain Question Answering), Kiểm chứng tính xác thực của thông tin (Fact Verification) hay "
        "sinh câu hỏi chuyên sâu — các mô hình chỉ dựa thuần túy vào bộ nhớ tham số bộc lộ 3 hạn chế cố hữu cực kỳ nghiêm trọng:"
    )
    
    add_bullet(doc, "Không thể cập nhật hoặc mở rộng tri thức (Inability to update/expand knowledge)", 
        "Tri thức của mô hình bị 'đóng băng' (frozen) tại thời điểm kết thúc quá trình huấn luyện. Để cập nhật một sự kiện mới diễn ra "
        "hoặc sửa đổi một thông tin đã lỗi thời/sai lệch, người ta buộc phải tái huấn luyện (retrain) hoặc tiếp tục fine-tune toàn bộ mô hình. "
        "Điều này tiêu tốn chi phí tính toán và tài nguyên phần cứng vô cùng lớn, đồng thời dễ dẫn đến hiện tượng 'quên thảm họa' (catastrophic forgetting)."
    )
    add_bullet(doc, "Hiện tượng ảo giác nghiêm trọng (Hallucination)", 
        "Khi đối mặt với các câu hỏi đòi hỏi thông tin chính xác, chi tiết hoặc các thực thể hiếm (rare entities/tail knowledge), mô hình "
        "thuần tham số thường tự ý bịa đặt (hallucinate) ra câu trả lời có vẻ rất hợp lý và ngữ pháp trôi chảy nhưng thực chất lại sai lệch hoàn toàn so với thực tế."
    )
    add_bullet(doc, "Thiếu khả năng truy vết và kiểm chứng nguồn tin (Lack of Provenance & Interpretability)", 
        "Mô hình hoạt động như một 'hộp đen' (black box). Khi sinh ra một câu khẳng định, nó hoàn toàn không thể cung cấp tài liệu dẫn chứng "
        "hoặc trích dẫn nguồn gốc đoạn văn cụ thể đã căn cứ vào đó, khiến hệ thống không đáp ứng được yêu cầu về độ tin cậy trong các lĩnh vực quan trọng như y tế, giáo dục hay pháp lý."
    )
    
    add_styled_paragraph(doc,
        "Để giải quyết triệt để bài toán này, nhóm tác giả đề xuất kiến trúc Retrieval-Augmented Generation (RAG) — một khung làm việc tổng quát "
        "kết hợp hài hòa giữa Bộ nhớ tham số (Parametric Memory - một mô hình Seq2Seq Transformer được tiền huấn luyện, cụ thể là BART) "
        "và Bộ nhớ phi tham số (Non-parametric Memory - một kho chỉ mục vector dày đặc chứa hơn 21 triệu đoạn văn bản Wikipedia trích xuất bằng Dense Passage Retrieval). "
        "Bài toán trọng tâm được đặt ra là: Làm thế nào để điều kiện hóa quá trình sinh ngôn ngữ của Seq2Seq Generator dựa trên các đoạn văn bản "
        "truy hồi được từ kho dữ liệu ngoài, đồng thời huấn luyện mô hình theo phương thức end-to-end mà không cần gán nhãn giám sát đoạn văn nào là phù hợp nhất cho từng câu hỏi."
    )

    add_heading_2(doc, "1.2. Các câu hỏi nghiên cứu của bài báo (Research Questions)")
    add_styled_paragraph(doc, 
        "Nhằm kiểm chứng tính khả thi và ưu thế vượt trội của kiến trúc RAG, bài báo thiết lập và giải quyết 4 câu hỏi nghiên cứu cốt lõi:"
    )
    add_bullet(doc, "Câu hỏi nghiên cứu 1 (RQ1 - Hiệu năng so với Baseline)", 
        "Liệu việc kết hợp một bộ truy hồi dày đặc phi tham số (Dense Passage Retriever - DPR) với một mô hình sinh chuỗi tham số (Seq2Seq BART) "
        "có thể vượt qua hiệu năng của các mô hình ngôn ngữ khổng lồ chỉ dùng tham số nội tại (như T5-11B với 11 tỷ tham số) cũng như các hệ thống "
        "hỏi đáp trích xuất hai giai đoạn (Extractive Retrieve-and-Read) trên các bộ benchmark Open-Domain QA hàng đầu hay không?"
    )
    add_bullet(doc, "Câu hỏi nghiên cứu 2 (RQ2 - Cơ chế biên giải RAG-Sequence vs RAG-Token)", 
        "Giữa hai công thức biên giải xác suất (marginalization): RAG-Sequence (chọn một tập văn bản cố định để dẫn hướng toàn bộ câu trả lời) "
        "và RAG-Token (cho phép mỗi token sinh ra có thể tham chiếu đến một phân phối tài liệu truy hồi khác nhau), mô hình nào đem lại hiệu quả "
        "vượt trội hơn đối với từng loại nhiệm vụ (câu trả lời ngắn gọn, có cấu trúc so với đoạn văn tự do, mở rộng)?"
    )
    add_bullet(doc, "Câu hỏi nghiên cứu 3 (RQ3 - Khả năng hoán đổi và cập nhật tri thức không cần train lại)", 
        "Khi thế giới thực thay đổi tri thức, liệu ta có thể chỉ cần 'thay thế nóng' (hot-swap) chỉ mục phi tham số (ví dụ thay kho Wikipedia năm 2016 "
        "bằng Wikipedia năm 2018) mà không cần huấn luyện lại bất kỳ tham số nào của bộ sinh (Generator) để cập nhật thông tin chuẩn xác cho mô hình hay không?"
    )
    add_bullet(doc, "Câu hỏi nghiên cứu 4 (RQ4 - Tính trung thực và giảm thiểu ảo giác)", 
        "Văn bản được tạo ra bởi RAG có thực sự đạt độ trung thực cao hơn (more factual), đa dạng hơn và giảm thiểu đáng kể hiện tượng bịa đặt thông tin "
        "(hallucination) so với các mô hình sinh văn bản truyền thống chỉ dựa vào bộ nhớ tham số hay không?"
    )

    # =========================================================================
    # CHƯƠNG 2
    # =========================================================================
    add_heading_1(doc, "CHƯƠNG 2: CÁC CÔNG TRÌNH LIÊN QUAN (RELATED WORK)")
    add_styled_paragraph(doc,
        "Bài báo của Lewis et al. (NeurIPS 2020) được xây dựng dựa trên sự kế thừa và đột phá từ 3 nhánh nghiên cứu lớn trong lịch sử phát triển của NLP:"
    )

    add_heading_2(doc, "2.1. Nhóm kiến trúc kết hợp bộ nhớ tham số và phi tham số (Hybrid Memory Models)")
    add_styled_paragraph(doc,
        "Ý tưởng mở rộng mô hình nơ-ron bằng bộ nhớ lưu trữ ngoài bắt đầu từ các kiến trúc Memory Networks (Weston et al., 2014; Sukhbaatar et al., 2015). "
        "Các mô hình này lưu các sự kiện hoặc câu văn vào một ma trận bộ nhớ và sử dụng cơ chế chú ý liên tục để truy xuất thông tin liên quan. "
        "Gần đây hơn, kiến trúc REALM (Guu et al., 2020) đã tạo ra một bước ngoặt khi lần đầu tiên huấn luyện một bộ truy hồi nơ-ron (Neural Retriever) "
        "đồng thời với một mô hình Masked Language Model (dựa trên BERT) theo phương pháp học không giám sát end-to-end. Tuy nhiên, REALM chỉ tập trung "
        "vào bài toán tiền huấn luyện (pre-training) và suy luận dạng trích xuất từ bị che khuất (masked token prediction), chưa có khả năng sinh chuỗi "
        "tự do hoàn chỉnh (Sequence-to-Sequence generation). Cùng thời điểm, kNN-LM (Khandelwal et al., 2020) đề xuất mở rộng mô hình ngôn ngữ bằng cách "
        "nội suy phân phối xác suất từ tiếp theo với các lân cận gần nhất trong không gian vector biểu diễn, chứng minh tiềm năng to lớn của việc truy hồi ngoài."
    )

    add_heading_2(doc, "2.2. Nhóm hệ thống hỏi-đáp Open-Domain truyền thống (Open-Domain Question Answering)")
    add_styled_paragraph(doc,
        "Trong các hệ thống hỏi đáp Open-Domain truyền thống, hướng tiếp cận phổ biến nhất là quy trình hai giai đoạn 'Truy hồi và Đọc' (Retrieve-and-Read), "
        "tiêu biểu là hệ thống DrQA (Chen et al., 2017) và ORQA (Lee et al., 2019). Giai đoạn đầu sử dụng các kỹ thuật đối sánh từ khóa thưa thớt "
        "(sparse keyword matching) như TF-IDF hoặc BM25 để lọc ra top-k bài viết có liên quan. Giai đoạn thứ hai sử dụng một mô hình đọc trích xuất "
        "(Extractive Reader) để xác định vị trí bắt đầu và kết thúc (span) của câu trả lời trong văn bản. Hạn chế cốt tử của phương pháp này là: "
        "(1) BM25 không hiểu được ngữ nghĩa sâu xa và từ đồng nghĩa; và (2) Extractive Reader chỉ trích xuất được những câu trả lời đã tồn tại nguyên văn "
        "trong một đoạn văn duy nhất, hoàn toàn bất lực trước các câu hỏi đòi hỏi phải tổng hợp thông tin từ nhiều nguồn hoặc diễn đạt lại theo cách tự nhiên."
    )
    add_styled_paragraph(doc,
        "Bước nhảy vọt quyết định xuất hiện khi Karpukhin et al. (EMNLP 2020) công bố Dense Passage Retrieval (DPR). DPR thay thế hoàn toàn BM25 bằng "
        "mạng hai nhánh (Dual-Encoder) gồm hai mô hình BERT độc lập: một để mã hóa câu hỏi và một để mã hóa đoạn văn. Được huấn luyện qua hàm mất mát "
        "tương phản (contrastive loss), DPR đạt khả năng truy hồi vượt xa BM25 ở các chỉ số Recall@k. Bài báo RAG đã trực tiếp kế thừa DPR làm thành phần "
        "truy hồi phi tham số nền tảng của mình."
    )

    add_heading_2(doc, "2.3. Nhóm mô hình sinh ngôn ngữ trực tiếp (Generative / Closed-book QA)")
    add_styled_paragraph(doc,
        "Ngược lại với hệ thống truy hồi, một nhánh nghiên cứu khác khai phá tiềm năng của mô hình 'Closed-book QA' (Roberts et al., 2020; Raffel et al., 2020). "
        "Các nhà nghiên cứu đưa trực tiếp câu hỏi vào mô hình T5 có quy mô khổng lồ (lên đến 11 tỷ tham số - T5-11B) và yêu cầu mô hình sinh câu trả lời "
        "mà không cung cấp bất kỳ văn bản tham khảo nào. Mặc dù T5-11B đạt kết quả đáng kinh ngạc, nó đòi hỏi tài nguyên huấn luyện siêu lớn và gặp hiện tượng "
        "ảo giác nghiêm trọng. Đồng thời, Fusion-in-Decoder (FiD) của Izacard & Grave (2020) cũng đề xuất giải pháp mã hóa từng đoạn văn độc lập bằng encoder "
        "rồi kết hợp chúng tại tầng cross-attention của decoder để sinh câu trả lời."
    )

    add_heading_2(doc, "2.4. Vị trí học thuật và Đóng góp mang tính cách mạng của RAG (NeurIPS 2020)")
    add_styled_paragraph(doc,
        "Mô hình RAG của Lewis et al. đã giải quyết toàn diện các nhược điểm của các phương pháp trên bằng cách kết hợp sức mạnh biểu diễn ngữ nghĩa của "
        "DPR với khả năng sinh ngôn ngữ tự nhiên xuất sắc của BART-large. Bảng dưới đây so sánh RAG với các hướng tiếp cận chính trong lịch sử NLP:"
    )

    # Table 2-1: Comparison
    t_comp = doc.add_table(rows=5, cols=5)
    t_comp.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers = ["Phương pháp", "Loại bộ nhớ", "Cơ chế truy hồi", "Khả năng sinh tự do", "Khả năng cập nhật tri thức"]
    for c_i, h in enumerate(headers):
        cell = t_comp.rows[0].cells[c_i]
        cell.text = h
        set_cell_background(cell, "003366")
        set_cell_margins(cell, top=100, bottom=100, left=100, right=100)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for r in p.runs:
            r.font.name = 'Times New Roman'
            r.font.size = Pt(10)
            r.font.bold = True
            r.font.color.rgb = RGBColor(255, 255, 255)
            
    rows_data = [
        ("DrQA (Chen et al., 2017)", "Chỉ phi tham số", "BM25 / TF-IDF", "Không (Chỉ trích xuất span)", "Dễ dàng (Cập nhật văn bản)"),
        ("T5-11B Closed-Book", "Thuần tham số (11B)", "Không có truy hồi", "Có (Sinh câu tự do)", "Rất khó (Phải retrain lại)"),
        ("REALM (Guu et al., 2020)", "Lai (Hybrid)", "Neural Dense", "Không (Chỉ đoán masked word)", "Có thể cập nhật"),
        ("RAG (Lewis et al., 2020)", "Lai (Hybrid: BART + DPR)", "Neural Dense (DPR)", "Có (Sinh câu tự do mượt mà)", "Hoán đổi nóng tức thì (Hot-swap)")
    ]
    for r_i, r_data in enumerate(rows_data):
        row = t_comp.rows[r_i + 1]
        bg = "F2F2F2" if r_i % 2 == 1 else "FFFFFF"
        for c_i, val in enumerate(r_data):
            cell = row.cells[c_i]
            cell.text = val
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=80, bottom=80, left=100, right=100)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_i != 0 else WD_ALIGN_PARAGRAPH.LEFT
            for r in p.runs:
                r.font.name = 'Times New Roman'
                r.font.size = Pt(10)

    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_cap = p_cap.add_run("Bảng 2-1: So sánh tổng quan giữa RAG và các phương pháp tiền nhiệm trong NLP")
    r_cap.font.name = 'Times New Roman'
    r_cap.font.size = Pt(10)
    r_cap.font.italic = True

    # =========================================================================
    # CHƯƠNG 3
    # =========================================================================
    add_heading_1(doc, "CHƯƠNG 3: DỮ LIỆU VÀ QUY TRÌNH TIỀN XỬ LÝ CỦA BÀI BÁO (DATASET & PREPROCESSING)")
    
    add_heading_2(doc, "3.1. Các tập dữ liệu khảo sát trong bài báo (Datasets Breakdown)")
    add_styled_paragraph(doc,
        "Để chứng minh tính tổng quát và sức mạnh vượt trội, bài báo NeurIPS 2020 đã thực nghiệm mô hình RAG trên 4 nhóm tác vụ khác nhau, "
        "trong đó trọng tâm lớn nhất đặt vào bài toán Hỏi - Đáp trong miền mở (Open-Domain Question Answering) với các tập dữ liệu tiêu chuẩn sau:"
    )
    
    add_bullet(doc, "1. Kho tri thức nguồn phi tham số (Non-parametric Knowledge Base - Wikipedia Dump)",
        "Toàn bộ tri thức thế giới được trích xuất từ bản dump tháng 12/2018 của bách khoa toàn thư Wikipedia tiếng Anh. "
        "Đây là kho dữ liệu khổng lồ bao trùm mọi chủ đề khoa học, lịch sử, văn hóa và đời sống."
    )
    add_bullet(doc, "2. Tập dữ liệu Natural Questions (NQ - Kwiatkowski et al., 2019)",
        "Được xem là tiêu chuẩn vàng trong đánh giá Open-Domain QA. Dữ liệu gồm các câu hỏi thực tế được người dùng gõ vào công cụ tìm kiếm Google. "
        "Các câu hỏi này mang tính tự nhiên, không bị định hướng hay gượng gạo. Nhãn chuẩn (Ground Truth) là các câu trả lời ngắn gọn được các chuyên gia "
        "tìm kiếm và xác nhận từ các bài viết Wikipedia liên quan. Tập dữ liệu gồm: 79.168 mẫu huấn luyện (train), 8.757 mẫu phát triển (dev) và 3.610 mẫu kiểm thử (test)."
    )
    add_bullet(doc, "3. Tập dữ liệu TriviaQA (Joshi et al., 2017)",
        "Tập hợp các câu đố trắc nghiệm và câu hỏi đố vui từ các cuộc thi trực tuyến do những người say mê đố vui biên soạn, đi kèm các bài báo Wikipedia tương ứng. "
        "Khác với NQ, câu hỏi của TriviaQA thường giàu thực thể, mang tính suy luận phức tạp và chứa đựng nhiều dữ kiện ngữ cảnh hơn. "
        "Tập dữ liệu bao gồm: 78.785 cặp câu hỏi-đáp trong tập train, 8.837 mẫu trong tập dev, và 11.313 mẫu kiểm thử độc lập (test split)."
    )
    add_bullet(doc, "4. Tập dữ liệu WebQuestions (WQ) & CuratedTREC (CT)",
        "WebQuestions (Berant et al., 2013) gồm các câu hỏi thu thập từ Google Suggest API với câu trả lời là các thực thể trong cơ sở tri thức Freebase (3.778 train, 2.032 test). "
        "CuratedTREC (Baudiš & Šedivý, 2015) là tập dữ liệu QA ngắn truyền thống được chuẩn hóa từ các cuộc thi TREC QA qua nhiều năm (1.484 mẫu)."
    )
    add_bullet(doc, "5. Các tác vụ chuyên sâu tri thức mở rộng khác",
        "Ngoài Open-Domain QA, bài báo còn thử nghiệm trên: MS-MARCO (sinh câu trả lời tự do - Abstractive QA), FEVER (bài toán phân loại 3 nhãn: Supported / Refuted / NotEnoughInfo "
        "để kiểm chứng sự thật), và Jeopardy Question Generation (sinh câu hỏi từ câu trả lời và điều kiện ngữ cảnh)."
    )

    # Table 3-1
    t_data = doc.add_table(rows=6, cols=6)
    t_data.alignment = WD_TABLE_ALIGNMENT.CENTER
    d_headers = ["Tập dữ liệu", "Nguồn câu hỏi", "Train Split", "Dev Split", "Test Split", "Đặc điểm nhãn"]
    for c_i, h in enumerate(d_headers):
        cell = t_data.rows[0].cells[c_i]
        cell.text = h
        set_cell_background(cell, "003366")
        set_cell_margins(cell, top=100, bottom=100, left=100, right=100)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for r in p.runs:
            r.font.name = 'Times New Roman'
            r.font.size = Pt(10)
            r.font.bold = True
            r.font.color.rgb = RGBColor(255, 255, 255)

    data_rows = [
        ("Natural Questions (NQ)", "Google Search thực tế", "79.168", "8.757", "3.610", "Câu trả lời ngắn (Short answer span)"),
        ("TriviaQA", "Diễn đàn đố vui trực tuyến", "78.785", "8.837", "11.313", "Thực thể / Từ khóa đố vui"),
        ("WebQuestions (WQ)", "Google Suggest API", "3.778", "-", "2.032", "Thực thể cơ sở tri thức Freebase"),
        ("CuratedTREC (CT)", "Cuộc thi TREC QA chuẩn", "1.484", "-", "694", "Dạng text span chuẩn hóa"),
        ("Wikipedia Dump (Corpus)", "Wikipedia (12/2018)", "21.015.324 passages", "-", "-", "100 từ / passage + Title")
    ]
    for r_i, r_data in enumerate(data_rows):
        row = t_data.rows[r_i + 1]
        bg = "F2F2F2" if r_i % 2 == 1 else "FFFFFF"
        for c_i, val in enumerate(r_data):
            cell = row.cells[c_i]
            cell.text = val
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=80, bottom=80, left=80, right=80)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_i in [2, 3, 4] else WD_ALIGN_PARAGRAPH.LEFT
            for r in p.runs:
                r.font.name = 'Times New Roman'
                r.font.size = Pt(10)

    p_cap2 = doc.add_paragraph()
    p_cap2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_cap2 = p_cap2.add_run("Bảng 3-1: Thống kê chi tiết các tập dữ liệu thực nghiệm trong bài báo Lewis et al. (NeurIPS 2020)")
    r_cap2.font.name = 'Times New Roman'
    r_cap2.font.size = Pt(10)
    r_cap2.font.italic = True

    add_heading_2(doc, "3.2. Quy trình Tiền xử lý dữ liệu chi tiết của bài báo (Data Preprocessing Pipeline)")
    add_styled_paragraph(doc,
        "Để biến một kho dữ liệu thô khổng lồ thành một hệ thống có khả năng truy hồi và sinh câu trả lời trong vài mili-giây, "
        "nhóm tác giả bài báo đã thiết kế một pipeline tiền xử lý chặt chẽ gồm 4 bước kỹ thuật chính:"
    )

    add_heading_3(doc, "Bước 1: Làm sạch và Phân đoạn văn bản Wikipedia (Wikipedia Chunking)")
    add_styled_paragraph(doc,
        "Từ bản dump Wikipedia tháng 12/2018 thô, tác giả tiến hành lọc bỏ toàn bộ các thẻ HTML, cú pháp Wiki markup, bảng biểu phức tạp, "
        "danh sách liên kết, chú thích ảnh và chỉ trích xuất phần văn bản thuần (plain text). "
        "Sau đó, văn bản của mỗi bài viết được chia nhỏ thành các đoạn văn độc lập (passages / chunks) theo nguyên tắc:"
    )
    add_bullet(doc, "Độ dài cố định 100 từ (Word-level splitting)",
        "Mỗi passage chứa đúng 100 từ liên tiếp. Điểm đặc biệt là bài báo áp dụng kỹ thuật phân rã không gối đầu (non-overlapping chunks) "
        "thay vì trượt cửa sổ (sliding window) như các nghiên cứu sau này, nhằm tiết kiệm không gian lưu trữ chỉ mục."
    )
    add_bullet(doc, "Gắn kèm Tiêu đề bài viết (Title Prepending)",
        "Để tránh việc đoạn văn bị mất ngữ cảnh chủ đề khi đứng biệt lập (ví dụ một đoạn văn chỉ ghi 'Ông sinh năm 1945 tại...' mà không nói rõ là ai), "
        "tác giả gắn tiêu đề bài viết vào đầu mỗi chunk theo định dạng chuẩn: 'Title: [Tên bài viết] | Passage: [100 từ nội dung]'. "
        "Kỹ thuật tiền xử lý này cải thiện đáng kể độ chính xác của vector biểu diễn khi nhúng vào không gian ngữ nghĩa."
    )
    add_styled_paragraph(doc,
        "➔ Kết quả của Bước 1: Tạo ra tổng cộng đúng 21.015.324 passages (hơn 21 triệu đoạn văn) được đánh mã định danh (id) duy nhất."
    )

    add_heading_3(doc, "Bước 2: Vector hóa và Đánh chỉ mục Dày đặc (Dense Indexing with FAISS)")
    add_styled_paragraph(doc,
        "Toàn bộ 21.015.324 đoạn văn được xử lý ngoại tuyến (offline) thông qua bộ mã hóa tài liệu của DPR (Passage Encoder - dựa trên kiến trúc BERT-base gồm 110M tham số). "
        "Mỗi đoạn văn được biểu diễn bằng vector nhúng tại vị trí token [CLS], tạo thành một vector thực 768 chiều. "
        "Để tìm kiếm lân cận gần đúng trong không gian 21 triệu vector 768 chiều ở thời gian thực, tác giả xây dựng chỉ mục tìm kiếm tích vô hướng cực đại "
        "(Maximum Inner Product Search - MIPS) bằng thư viện FAISS (Facebook AI Similarity Search):"
    )
    add_bullet(doc, "Cấu trúc chỉ mục HNSW (Hierarchical Navigable Small World)",
        "FAISS sử dụng cấu trúc đồ thị đa tầng HNSW. Phương pháp này cân bằng hoàn hảo giữa độ chính xác truy hồi (Recall) và tốc độ tìm kiếm: "
        "cho phép tìm ra top-k (thường là top 5 đến top 100) đoạn văn có tích vô hướng cao nhất với vector câu hỏi chỉ trong vòng dưới 15 mili-giây."
    )

    add_heading_3(doc, "Bước 3: Tiền xử lý câu hỏi truy vấn (Query Processing)")
    add_styled_paragraph(doc,
        "Khác với Information Retrieval truyền thống (cần loại bỏ stop words, stemming từ ngữ), đối với câu hỏi đầu vào x:"
    )
    add_bullet(doc, "Bảo toàn nguyên vẹn cấu trúc câu hỏi tự nhiên",
        "Câu hỏi được giữ nguyên toàn bộ các từ chức năng và cấu trúc ngữ pháp, vì mô hình ngôn ngữ Transformer cần toàn bộ bối cảnh để nắm bắt ý định sâu xa của người dùng."
    )
    add_bullet(doc, "Mã hóa câu hỏi thời gian thực (Online Query Encoding)",
        "Câu hỏi x được đưa qua DPR Question Encoder (BERT-base, 768 chiều) để tạo ra vector truy vấn q(x). "
        "Hệ thống FAISS sau đó tính tích vô hướng d(z)^T * q(x) với toàn bộ 21 triệu vector passages để truy xuất top-k văn bản liên quan nhất."
    )

    add_heading_3(doc, "Bước 4: Chuẩn hóa đầu vào cho bộ sinh Seq2Seq (Input Formatting for BART Generator)")
    add_styled_paragraph(doc,
        "Sau khi truy hồi được tập tài liệu {z_1, z_2, ..., z_k}, để đưa vào bộ sinh BART-large, bài báo áp dụng quy tắc ghép chuỗi chuẩn hóa:"
    )
    add_bullet(doc, "Ghép nối Câu hỏi và Đoạn văn (Concatenation)",
        "Với mỗi đoạn văn z_i, một chuỗi đầu vào được tạo ra theo khuôn mẫu đặc biệt của BART: '<s> question: [Câu hỏi x] </s> document: [Tiêu đề] [Đoạn văn z_i] </s>'. "
        "Các token đặc biệt <s> và </s> đóng vai trò phân định rõ ràng ranh giới giữa câu hỏi của người dùng và tài liệu dẫn chứng."
    )
    add_bullet(doc, "Token hóa Byte-Pair Encoding (BPE Tokenization)",
        "Chuỗi ghép nối trên được token hóa bằng bộ tokenizer Byte-Pair Encoding chuẩn của BART với không gian từ vựng gồm 50.265 subwords. "
        "Độ dài chuỗi đầu vào được cắt bớt (truncation) ở ngưỡng tối đa để bảo đảm không vượt quá giới hạn ngữ cảnh của BART và tối ưu hóa bộ nhớ GPU khi huấn luyện."
    )

    # =========================================================================
    # CHƯƠNG 4
    # =========================================================================
    add_heading_1(doc, "CHƯƠNG 4: KẾT LUẬN TUẦN 1 VÀ KẾ HOẠCH TUẦN 2 (25/9 – 3/10)")
    
    add_heading_2(doc, "4.1. Tổng kết những hiểu biết đạt được từ bài báo trong Tuần 1")
    add_styled_paragraph(doc,
        "Trong tuần làm việc đầu tiên (19/09 – 25/09/2026), nhóm nghiên cứu đã bám sát yêu cầu đề bài và hoàn thành xuất sắc các mục tiêu học thuật từ bài báo nền tảng Lewis et al. (NeurIPS 2020):"
    )
    add_bullet(doc, "Về mặt bài toán (Problem)", 
        "Làm rõ nguyên nhân cốt lõi khiến các mô hình ngôn ngữ thuần tham số thất bại trong các bài toán Knowledge-Intensive (thiếu cập nhật, ảo giác, thiếu nguồn tin) "
        "và hiểu sâu sắc kiến trúc lai ghép RAG kết hợp Parametric Generator với Non-parametric Dense Memory."
    )
    add_bullet(doc, "Về câu hỏi nghiên cứu (Research Questions)", 
        "Xác lập rõ ràng 4 câu hỏi nghiên cứu (RQ1 – RQ4), nắm được bản chất xác suất của hai biến thể RAG-Sequence và RAG-Token trong việc biên giải phân phối câu trả lời."
    )
    add_bullet(doc, "Về công trình liên quan (Related Work)", 
        "Hệ thống hóa được sự tiến hóa từ Memory Networks, DrQA (BM25), REALM đến DPR và FiD; định vị được đóng góp mang tính bước ngoặt của RAG trên bản đồ nghiên cứu NLP thế giới."
    )
    add_bullet(doc, "Về dữ liệu và tiền xử lý (Dataset & Preprocessing)", 
        "Nắm vững đặc thù của hai tập dữ liệu benchmark Natural Questions và TriviaQA, đồng thời giải mã chi tiết toàn bộ pipeline tiền xử lý 4 bước của bài báo "
        "(tách đoạn Wikipedia 100 từ có gắn Title, mã hóa DPR 768 chiều, đánh chỉ mục FAISS HNSW MIPS, và định dạng chuỗi ghép BPE cho BART)."
    )

    add_heading_2(doc, "4.2. Kế hoạch hành động cụ thể cho Tuần 2 (25/09 – 03/10/2026)")
    add_styled_paragraph(doc,
        "Căn cứ theo kế hoạch chung của học phần Xử lý Ngôn ngữ Tự nhiên ghi trong sổ theo dõi tiến độ đồ án, trong Tuần 2, nhóm sẽ chuyển từ giai đoạn "
        "nghiên cứu lý thuyết bài báo sang giai đoạn thực nghiệm tiền xử lý dữ liệu thực tế:"
    )
    add_bullet(doc, "Nhiệm vụ 1 (Tải và thiết lập tập dữ liệu con - Subset Selection)", 
        "Thu thập và tải tập dữ liệu con (subset khoảng 800 - 1000 mẫu) từ Natural Questions hoặc TriviaQA công khai trên Hugging Face Datasets "
        "để bảo đảm khả năng tái lập và huấn luyện ổn định trên môi trường phần cứng kiểm soát."
    )
    add_bullet(doc, "Nhiệm vụ 2 (Xây dựng Pipeline Tiền xử lý dữ liệu)", 
        "Lập trình đầy đủ các hàm làm sạch văn bản (text cleaning), chuẩn hóa ký tự lạ (normalization), phân đoạn văn bản thành các chunks có độ dài cố định kèm overlap, "
        "và phân chia tập dữ liệu train/dev/test với tỷ lệ cố định (ví dụ 85% train - 15% test) sử dụng random seed=42 để bảo đảm tính khách quan."
    )
    add_bullet(doc, "Nhiệm vụ 3 (Kiểm tra chất lượng và Phòng chống rò rỉ dữ liệu)", 
        "Thực hiện kiểm tra hiện tượng mất cân bằng lớp (data imbalance), rà soát dữ liệu rác/thiếu (noisy/missing data), và đặc biệt kiểm soát chặt chẽ "
        "để không xảy ra rò rỉ dữ liệu (data leakage) giữa tập văn bản ngữ liệu với tập câu hỏi kiểm thử."
    )
    add_bullet(doc, "Nhiệm vụ 4 (Lập báo cáo và Notebook Preprocessing)", 
        "Đóng gói toàn bộ mã nguồn tiền xử lý vào Jupyter Notebook hoàn chỉnh, xuất ra bảng thống kê số lượng văn bản, số lượng cặp câu hỏi-đáp, "
        "phân phối độ dài từ, và viết báo cáo tiến độ Tuần 2."
    )

    # =========================================================================
    # TÀI LIỆU THAM KHẢO
    # =========================================================================
    add_heading_1(doc, "TÀI LIỆU THAM KHẢO (REFERENCES)")
    refs = [
        "[1] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. In Advances in Neural Information Processing Systems (NeurIPS 2020), 33, 9459-9474.",
        "[2] Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, Wen-tau Yih (2020). Dense Passage Retrieval for Open-Domain Question Answering. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP 2020), pages 6769-6781.",
        "[3] Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Veselin Stoyanov, Luke Zettlemoyer (2020). BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL 2020), pages 7871-7880.",
        "[4] Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, Ming-Wei Chang (2020). REALM: Retrieval-Augmented Language Model Pre-training. In International Conference on Machine Learning (ICML 2020), PMLR 119:3929-3938.",
        "[5] Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, Slav Petrov (2019). Natural Questions: A Benchmark for Question Answering Research. Transactions of the Association for Computational Linguistics (TACL), 7:452-466.",
        "[6] Mandar Joshi, Eunsol Choi, Daniel S. Weld, Luke Zettlemoyer (2017). TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1601-1611.",
        "[7] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu (2020). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. Journal of Machine Learning Research (JMLR), 21(140):1-67.",
        "[8] Danqi Chen, Adam Fisch, Jason Weston, Antoine Bordes (2017). Reading Wikipedia to Answer Open-Domain Questions. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (ACL 2017), pages 1870-1879."
    ]
    for ref in refs:
        p_ref = doc.add_paragraph(style='Normal')
        p_ref.paragraph_format.space_before = Pt(2)
        p_ref.paragraph_format.space_after = Pt(4)
        p_ref.paragraph_format.left_indent = Inches(0.25)
        p_ref.paragraph_format.first_line_indent = Inches(-0.25)
        r = p_ref.add_run(ref)
        r.font.name = 'Times New Roman'
        r.font.size = Pt(11)

    print(f"Đang lưu file báo cáo ra: {OUTPUT_PATH}...")
    doc.save(OUTPUT_PATH)
    print("Đã lưu thành công!")
    
    print(f"Sao chép thêm một bản vào thư mục dự án: {LOCAL_COPY_PATH}...")
    shutil.copyfile(OUTPUT_PATH, LOCAL_COPY_PATH)
    print("Đã sao chép hoàn tất!")

if __name__ == "__main__":
    create_report()
