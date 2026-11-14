# Báo cáo Day 7 — Phạm Hồng Khánh Huy (`2A202603588`)

## Dữ liệu và benchmark

- **Nguồn, quyền sử dụng, số trang và số chunk:**
  - Nguồn: Sách giáo trình mở OpenStax *Introduction to Computer Science* (`Introduction_To_Computer_Science_-_WEB.pdf`, giấy phép CC BY / OpenStax).
  - Phạm vi corpus (kế thừa Day 6): **34 trang nội dung hữu ích** (trang vật lý `19–52`, gồm Chương 1 và mở đầu Chương 2), được tách bằng chiến lược `B_paragraph` (`max_size=800`) thành **171 chunks** (`data/chunks.json`).
- **Vector store, embedding model, phiên bản và số chiều:**
  - Vector store: PostgreSQL 16 + `pgvector` (`pgvector/pgvector:0.8.7-pg16`, Docker local tại `127.0.0.1:55433`, database `day6_lab`, bảng `document_chunks`, khóa chính `(strategy, chunk_id)`, chỉ mục GIN trên `metadata JSONB`).
  - Embedding model: OpenAI `text-embedding-3-small`, **1536 chiều**, metric cosine similarity `1 - (embedding <=> query_vector)`.
- **Agent model, cách chạy và phân chia dev/test:**
  - Agent model: OpenAI `gpt-4o-mini` dùng tool calling (`src/agent.py`, `src/eval_agent.py`).
  - Cách chạy: Kích hoạt `.venv`, bật container PostgreSQL Day 6, chạy các lệnh trong `src/README.md` (`unittest`, `src/eval_retrieval.py`, `src/eval_agent.py`, `starter/agent_metrics.py`).
  - Phân chia benchmark (`data/questions.json`, tổng **24 câu**):
    - `dev` (17 câu): **15 câu có đáp án** (`q01`–`q15`) và **2 câu thiếu bằng chứng** (`q21`, `q22`).
    - `test` (7 câu): **5 câu có đáp án** (`q16`–`q20`) và **2 câu thiếu bằng chứng** (`q23`, `q24`).
- **Cách xác định câu có/thiếu evidence và tiêu chí chấm:**
  - Câu có đáp án: Có `gold` (`source_id`, `page`, `evidence`) đối chiếu trực tiếp trong `data/chunks.json` (20/20 câu khớp 100% sau chuẩn hóa khoảng trắng `\s+` do ngắt dòng PDF).
  - Câu thiếu bằng chứng: Câu hỏi gần chủ đề (nhắc tên DALL-E ở trang 48, Chương 13, Chương 14 ở trang 41, hoặc kiến thức Python ngoài phạm vi trang `19–52`), được kiểm tra thủ công toàn bộ 171 chunks để xác nhận không chứa đáp án.
  - Tiêu chí chọn cấu hình trên `dev` (chốt trước khi chạy `test`): Ưu tiên **Context Evidence Rate** và **Evidence Hit@3**, sau đó tới **Hit@1** và đánh đổi thời gian/chi phí.

## Retrieval và thử nghiệm

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| Baseline | `tfidf`, `top_k=3`, `filter=q['filter']`, không rerank, `budget=800` | Mốc lexical ban đầu với budget mặc định của starter | dev | 11/15 (73.3%) | 14/15 (93.3%) | 1/15 (6.7%) | 0.35 ms / \$0 |
| Cải thiện 1 | `tfidf`, `top_k=3`, `filter=q['filter']`, không rerank, `budget=3000` | Tăng `budget` từ 800 lên 3000 để các chunk dài (~638–799 ký tự + tiêu đề) không bị loại | dev | 11/15 (73.3%) | 14/15 (93.3%) | 14/15 (93.3%) | 0.36 ms / \$0 |
| Phương pháp 2 | `dense` (Day 6 `pgvector` `B_paragraph`), `top_k=3`, SQL GIN filter, `budget=3000` | Tìm kiếm ngữ nghĩa bằng vector `text-embedding-3-small` | dev | 7/15 (46.7%) | 14/15 (93.3%) | 14/15 (93.3%) | 259.44 ms / \$0.000006 |
| Phương pháp 3 | `hybrid_rrf` (`dense` + `tfidf` qua RRF `k=60`, `candidate_k=10`), `top_k=3`, `budget=3000` | Kết hợp khớp từ khóa chính xác của TF-IDF và ngữ nghĩa của Dense để tăng Hit@1 | dev | 11/15 (73.3%) | 14/15 (93.3%) | 14/15 (93.3%) | 293.21 ms / \$0.000006 |
| Cải thiện 2 | `hybrid_rrf_rerank` (`hybrid_rrf` lấy 10 ứng viên + LLM rerank `gpt-4o-mini`), `top_k=3`, `budget=3000` | Dùng LLM xếp hạng lại 10 ứng viên từ Hybrid để cứu các câu bị tụt khỏi top-3 | dev | 13/15 (86.7%) | 15/15 (100.0%) | 15/15 (100.0%) | 4128.48 ms / \$0.005231 |
| Cấu hình đã chọn | `hybrid_rrf_rerank`, `candidate_k=10`, `top_k=3`, `budget=3000` | Chạy đánh giá độc lập trên `test` sau khi chốt từ `dev` | test | 5/5 (100.0%) | 5/5 (100.0%) | 5/5 (100.0%) | 2969.71 ms / \$0.001738 |
| Baseline | `tfidf`, `top_k=3`, `budget=800` | Mốc baseline trên `test` | test | 4/5 (80.0%) | 5/5 (100.0%) | 1/5 (20.0%) | 0.29 ms / \$0 |

**Phân tích hai lỗi trên `dev` và lý do chọn cấu hình:**
1. **Lỗi 1 — Context Budget quá nhỏ (`budget=800` trên `dev-tfidf-b800`):** Ở câu `q01` (*"How does the textbook define an algorithm in Section 1.1?"*), TF-IDF tìm đúng `p20_paragraph_1` ở hạng 1, nhưng `pack_context` ghép tiêu đề `[id | source | page]` (~108 ký tự) cộng với text chunk khiến tổng chiều dài vượt 800 ký tự, dẫn đến loại bỏ chunk hạng 1 và chỉ giữ chunk hạng 2 (`p19_paragraph_0`, sai evidence). Khi tăng `budget=3000`, `Context Evidence Rate` tăng từ **1/15 (6.7%)** lên **14/15 (93.3%)**.
2. **Lỗi 2 — Hụt xếp hạng top-3 ở `q01` và `q11`:** `dense` đánh mất `q01` khỏi top-3 (do nhiều đoạn ở phần Introduction cũng nói về thuật toán), còn `tfidf` và `hybrid_rrf` đều hụt `q11` (*"What is information science according to Section 1.2?"*, trang 33) khỏi top-3. Khi mở rộng lấy 10 ứng viên từ Hybrid rồi rerank bằng `gpt-4o-mini` (`hybrid_rrf_rerank`), cả `Hit@1` (lên **86.7%**), `Hit@3` và `Context Evidence Rate` (đạt **15/15 = 100.0%**) đều được giải quyết.
3. **Lý do chọn cấu hình:** Chọn **`hybrid_rrf_rerank` (`candidate_k=10`, `top_k=3`, `budget=3000`)** làm cấu hình cuối vì đạt 100% `Evidence Hit@3` và `Context Evidence Rate` trên `dev`, với chi phí rất nhỏ (~0.00035 USD/query) và độ trễ chấp nhận được (~3–4 giây/query).

## Agent và đánh giá

- **Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi:**
  - Tool gọi bởi `gpt-4o-mini` (`src/agent.py`): `search_documents(query)` (gọi retriever đã cấu hình), `submit_answer(answer, citations)` và `abstain(reason)`.
  - Điều kiện dừng: Tối đa `max_searches=3` lượt tìm kiếm; ở lượt 0 bắt buộc gọi `search_documents`; các lượt sau model tự chọn tìm thêm với query viết lại, trả lời kèm citation hoặc từ chối khi thiếu bằng chứng. Nếu chạm giới hạn tìm kiếm hoặc citation không khớp chunk đã nằm trong context, agent từ chối (`status="abstain"`).
  - Chọn context: Deduplicate theo `chunk_id`, sắp xếp theo điểm retrieval giảm dần (`context_mode="score_desc"`) để các chunk điểm cao ở lượt tìm kiếm sau không bị các chunk lượt đầu chiếm hết `budget=3000` (khắc phục điểm yếu FIFO của khung starter). Baseline giữ nguyên `budget=800` và `context_mode="fifo"`.
  - Xử lý lỗi & an toàn: Prompt hệ thống quy định rõ nội dung tài liệu chỉ là **dữ liệu**, bỏ qua mọi chỉ dẫn nếu có trong văn bản; lỗi API/database được bắt riêng vào `status="error"` (không thay bằng kết quả giả).
  - *(Lưu ý debug trên `dev`: lần chạy đầu `agent-dev-hybrid` bị từ chối nhầm 12/15 câu vì model truyền ID vào tham số `citations` của `submit_answer` nhưng không lặp `[chunk-id]` trong text câu trả lời; đã sửa `src/agent.py` tự chuẩn hóa nối `[chunk-id]` nếu ID hợp lệ đã nằm trong `context_chunks`.)*

| Cấu hình / split | Tool use | Số lượt gọi (TB) | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---:|---|---|---|---|---|
| `agent-dev-hybrid` (`hybrid`, `budget=3000`, `dev`) | 17/17 | 1.35 | *(Chạy thử nghiệm trên dev: 14/15 trả lời có citation, chỉ hụt `q01`)* | — | 14/14 | 2/2 | 1/15 (`q01`) |
| `agent-test-baseline` (`tfidf`, `budget=800`, `fifo`, `test`) | 7/7 | 2.43 | 0: 4, 1: 0, 2: 1 · TB **0.40/2** | 1/1 (100%) | 1/1 | 2/2 | 4/5 (`q16`, `q18`, `q19`, `q20`) |
| `agent-test-selected` (`hybrid_rrf_rerank`, `budget=3000`, `score_desc`, `test`) | 7/7 | 1.57 | 0: 0, 1: 0, 2: 5 · TB **2.00/2** | 8/8 (100%) | 5/5 | 2/2 | 0/5 |

**Bảng đối chiếu chi tiết từng câu `test` (`agent-test-selected`):**

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| `q16` | Có đáp án | 1 lượt `search_documents("computer science term popularized 1961 definition")` → trúng `p22_paragraph_1` | Nêu George E. Forsythe (1961), trích đúng định nghĩa và phạm vi con người quanh máy tính `[Introduction_To_Computer_Science_-_WEB_p22_paragraph_1]` | `answer_grade=2`, `citation_support=[true]`: chunk trang 22 hỗ trợ đầy đủ mọi ý. |
| `q17` | Có đáp án | 1 lượt `search_documents("Turing-complete definition")` → trúng `p24_paragraph_1..3` | Giải thích Turing-complete (chạy mọi thuật toán, phân biệt máy tính bỏ túi, lưu dữ liệu/chương trình trong bộ nhớ, nền tảng theoretical CS) `[...p24_paragraph_1][...p24_paragraph_2][...p24_paragraph_3]` | `answer_grade=2`, `citation_support=[true, true, true]`: cả 3 chunk trang 24 trực tiếp hỗ trợ từng phần câu trả lời. |
| `q18` | Có đáp án | 1 lượt `search_documents("technological fix problems conditions")` → trúng `p41_paragraph_5` | Nêu rõ technological fix gây nhiều vấn đề hơn khi áp dụng thiếu trách nhiệm, không cân nhắc bối cảnh xã hội/chính trị và chọn sai công nghệ `[...p41_paragraph_5]` | `answer_grade=2`, `citation_support=[true]`: `p41_paragraph_5` chứng minh đầy đủ điều kiện này. |
| `q19` | Có đáp án (có `filter`) | 1 lượt `search_documents("Section 2.1 Computational Thinking three As")` + `filter={"section": "2.1 Computational Thinking"}` → trúng `p52_paragraph_2` | Liệt kê đúng 3 pha: Abstraction (formulation), Automation (expressing solution), Analysis (execution & evaluation) `[...p52_paragraph_2]` | `answer_grade=2`, `citation_support=[true]`: khớp nguyên văn `p52_paragraph_2` trang 52. |
| `q20` | Có đáp án | 1 lượt `search_documents("social determination of technology")` → trúng `p38_paragraph_0`, `p38_paragraph_2` | Định nghĩa công nghệ vốn trung tính (con người quyết định tốt/xấu) và bổ sung tranh luận về AI bias từ dữ liệu `[...p38_paragraph_0][...p38_paragraph_2]` | `answer_grade=2`, `citation_support=[true, true]`: cả 2 chunk trang 38 hỗ trợ trực tiếp định nghĩa và ý bổ sung. |
| `q23` | Thiếu bằng chứng | 3 lượt `search_documents` về hyperparameter (learning rate, batch size) của DALL-E → không có trong context | Từ chối trả lời (`abstain`), nêu rõ tài liệu truy xuất không chứa thông số learning rate/batch size của DALL-E | `abstention_correct=true`: từ chối đúng, không suy diễn ngoài tài liệu. |
| `q24` | Thiếu bằng chứng | 3 lượt `search_documents` về 5 framework an ninh mạng ở Chương 14 → không có trong corpus | Từ chối trả lời (`abstain`), nêu rõ không có thông tin về 5 framework ở Chương 14 | `abstention_correct=true`: từ chối đúng (Chương 14 nằm ngoài trang 19–52). |

- **Đường dẫn trace, nhãn, lệnh chạy, thời gian và chi phí:**
  - Retrieval `dev`: `results/dev-tfidf-b800/`, `results/dev-tfidf-b3000/`, `results/dev-dense-b3000/`, `results/dev-hybrid-b3000/`, `results/dev-hybrid-rerank-b3000/`.
  - Retrieval `test`: `results/test-tfidf-b800/`, `results/test-hybrid-rerank-b3000/`.
  - Agent `dev` & `test`: `results/agent-dev-hybrid/`, `results/agent-test-baseline/` (0 lỗi, 2.43 lượt search TB, 4097.86 ms trung vị, \$0.00234), `results/agent-test-selected/` (0 lỗi, 1.57 lượt search TB, 7532.98 ms trung vị, \$0.006707).
  - Lệnh tổng hợp metrics chính thức:
    - `.\.venv\Scripts\python.exe starter/agent_metrics.py --cases results/agent-test-baseline/cases.json --traces results/agent-test-baseline/traces.json --reviews results/agent-test-baseline/reviews.json --split test --output results/agent-test-baseline/metrics.json`
    - `.\.venv\Scripts\python.exe starter/agent_metrics.py --cases results/agent-test-selected/cases.json --traces results/agent-test-selected/traces.json --reviews results/agent-test-selected/reviews.json --split test --output results/agent-test-selected/metrics.json`
- **Kết quả trước/sau, ví dụ lỗi, đánh đổi và giới hạn:**
  - So với baseline (`tfidf`, `budget=800`, `fifo`), cấu hình đã chọn (`hybrid_rrf_rerank`, `budget=3000`, `score_desc`) nâng **Context Evidence Rate** trên `test` từ `1/5 (20.0%)` lên `5/5 (100.0%)`, nâng **Grounded answer mean** từ `0.40/2` lên `2.00/2`, giảm **từ chối nhầm** từ `4/5` xuống `0/5`, đồng thời vẫn giữ **Abstention đúng** ở mức `2/2 (100%)`.
  - **Đánh đổi:** Thêm bước LLM rerank làm thời gian trung vị mỗi câu Agent tăng từ ~4.1 giây lên ~7.5 giây và chi phí cho 7 câu test tăng từ \$0.00234 lên \$0.00671.
  - **Giới hạn:** Tập `test` có kích thước tối thiểu (7 câu) và chỉ nằm trong 34 trang của một giáo trình tiếng Anh; trong `q17` model diễn giải nhẹ cụm *"can simulate a Turing machine"* dù ý chính vẫn khớp `p24_paragraph_1`; các câu dạng bảng/biểu đồ phức tạp chưa được thử nghiệm vì không thực hiện phần nâng cao.
