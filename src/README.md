# Source Code — Day 7 Agentic RAG

- `retrievers.py`: Cài đặt các phương pháp tìm kiếm:
  - `TfidfRetriever`: Lexical retrieval (kế thừa `starter/baseline.py`, hỗ trợ metadata filter).
  - `PgVectorDenseRetriever`: Dense retrieval kết nối PostgreSQL + `pgvector` từ Day 6 (`strategy='B_paragraph'`, embedding `text-embedding-3-small`, 1536 chiều).
  - `HybridRRFRetriever`: Kết hợp Dense + TF-IDF bằng Reciprocal Rank Fusion (RRF).
  - `LLMRerankRetriever`: Lấy `candidate_k` đoạn ứng viên rồi xếp hạng lại bằng LLM (`gpt-4o-mini`).
- `eval_retrieval.py`: Benchmark retrieval & context (`Hit@1`, `Hit@3`, `Context Evidence Rate`, độ trễ, chi phí) trên `dev` / `test`.
- `agent.py`: Vòng lặp Agentic RAG dùng OpenAI tool calling (`search_documents`, `submit_answer`, `abstain`) với deduplication và `context_mode` (`score_desc` / `fifo`).
- `eval_agent.py`: Chạy Agentic RAG trên `dev` / `test`, xuất `traces.json`, `cases.json`, `reviews_template.json` và `run_summary.json`.

## Cách chạy nhanh (từ thư mục gốc repo)

```powershell
# 1. Chạy unit tests
.\.venv\Scripts\python.exe -m unittest discover -s tests -v

# 2. Benchmark retrieval trên dev
.\.venv\Scripts\python.exe src/eval_retrieval.py --split dev --method tfidf --budget 800 --output results/dev-tfidf-b800
.\.venv\Scripts\python.exe src/eval_retrieval.py --split dev --method tfidf --budget 3000 --output results/dev-tfidf-b3000
.\.venv\Scripts\python.exe src/eval_retrieval.py --split dev --method dense --budget 3000 --output results/dev-dense-b3000
.\.venv\Scripts\python.exe src/eval_retrieval.py --split dev --method hybrid --budget 3000 --output results/dev-hybrid-b3000
.\.venv\Scripts\python.exe src/eval_retrieval.py --split dev --method hybrid_rerank --candidate-k 10 --budget 3000 --output results/dev-hybrid-rerank-b3000

# 3. Chạy Agent trên dev / test
.\.venv\Scripts\python.exe src/eval_agent.py --split dev --method hybrid --budget 3000 --output results/agent-dev-hybrid
```
