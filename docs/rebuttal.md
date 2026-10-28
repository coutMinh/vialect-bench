# Bàn giao codebase cho rebuttal

Nhánh làm việc: `rebuttal/cleanup`, tách từ `main` tại `1d09b07`.
Việc dọn repo không sửa lịch sử hoặc commit của `main`. Các thay đổi chưa commit
có sẵn trong working tree được mang sang nhánh mới và giữ lại.

## Những gì cần push

| Nhóm | File/thư mục giữ lại | Vai trò |
| --- | --- | --- |
| Benchmark chạy inference | `data/Finalized/probe_dialects.jsonl` | 400 case, 100/task; mỗi case có standard và đủ 6 dialect |
| Annotation finalized | `data/vialectbench_finalized_6.json` | 2.400 rewrite; đầu vào EDA, intrinsic, lexical và conversion |
| Mẫu nguồn | `data/{sentiment,nli,qa,mcqa}_100.jsonl` | Kiểm chứng nguồn, label, context, answer và sampling; không chỉ giữ probe |
| Sampling/qualification | `data/DIALECT_LEXICON_v2.xlsx`, `data/selection_summary.csv`, `data/qualification_10.jsonl` | Lexicon, provenance và ví dụ qualification |
| Inference | `src/`, `configs/`, `scripts/run_remaining_hf_probes.sh` | Prompt, parser, model settings và runner; gồm runner riêng cho Gemma 3 |
| Paper analysis | `notebooks/EDA.ipynb` | Paired metrics, confidence, bảng/hình và export hard cases từ output đã chạy |
| Data audit | `notebooks/probe_dialects_quality_audit.ipynb` | Đối chiếu finalized, probe và mẫu nguồn; kiểm tra cấu trúc MCQA |
| Normalization tùy chọn | `notebooks/ViDia2std_Normalizer.ipynb` | Giữ một bản phục vụ câu hỏi normalization; cần cấu hình môi trường/path riêng |
| Kết quả chính | `outputs/direct_prompting/*.jsonl` | 10 model × 2.800 dòng; gồm prediction và raw output, tránh chạy inference lại |
| Normalizer rescue | `outputs/normalizer_rescue/`, `outputs/normalized_direct_prompting/` | 541 input dialect đã chuẩn hóa và 753 prediction cho hard-failure cohort của 10 model |
| Normalizer full | `outputs/normalizer_full/`, `outputs/normalized_direct_prompting_full/` | Toàn bộ 2.400 dialect variant, cùng setting 10 model gốc; lưu normalized text, input và raw LLM output |
| Intrinsic cache | `outputs/finalized_6_complete_six_intrinsic_analysis/` | PPL/NLL cho đủ 400 source × 6 dialect, metadata và summary |
| Lexical analysis | `outputs/lexical_substitution_analysis/` | Alignment, review sheets đã điền, substitution records, swap NLL, bảng và hình |
| Code phân tích | `scripts/analyze_vialectbench_data.py`, `scripts/lexical_data_prep.py`, `scripts/lexical_swap_nll.py`, `scripts/lexical_visualize.py`, `scripts/utils.py` | Intrinsic và phân tích biến đổi ngôn ngữ; `utils.py` là dependency của lexical prep |
| Paper artifacts | `paper_artifacts/dataset_analysis/`, `paper_artifacts/direct_prompting/` | Giữ CSV/JSONL/TeX/PDF/PNG hiện có để đối chiếu và viết rebuttal |
| Tài liệu | `submission.pdf`, `VialectBench - Rebuttal Phase - AAAI 2027.pdf` | Bản nộp và danh sách câu hỏi/ghi chú rebuttal |
| Cài đặt/kiểm chứng | `requirements*.txt`, `.env.example`, `scripts/check_rebuttal_bundle.py`, `docs/rebuttal_manifest.json` | Môi trường phân tích nhẹ và kiểm tra snapshot dữ liệu/kết quả |

`.gitignore` có ngoại lệ rõ ràng cho các nhóm data, outputs và artifact cần chia sẻ.
EDA đọc kết quả rescue mới từ `outputs/normalized_direct_prompting/`; đây chỉ là
cohort hard-failure, không phải inference trên đủ 2.400 rewrite. Không đổi tên hoặc
di chuyển các thư mục cache hiện tại vì notebook/script đang đọc trực tiếp theo
đường dẫn đó.
Baseline full được phân tích riêng bằng `scripts/analyze_normalizer_full.py`;
không trộn mẫu rescue với mẫu full khi viết số liệu rebuttal.
Ghi chú phương pháp, text audit và case study cho Q3 nằm trong
`docs/normalizer_full_rebuttal.md`.

## Những gì bỏ khỏi nhánh chia sẻ

Các file đã tracked dưới đây được bỏ khỏi Git index bằng `git rm --cached` và
ignore. Sau đó, 254 file cũ/cache được chuyển vào
`.local/rebuttal-cleanup/legacy/`, giữ cấu trúc đường dẫn ban đầu để có thể khôi phục.
Cây thư mục làm việc chính chỉ còn phần phục vụ rebuttal.

| Nhóm bỏ | Lý do |
| --- | --- |
| `notebooks/seas.ipynb`, `paper_artifacts/seas/` | Side project SEAS |
| `VialectBench.github.io/` | Website riêng, không phải dependency của benchmark |
| `notebooks/ViDia2std_Normalizer (2).ipynb` | Bản trùng notebook normalizer |
| `paper_artifacts/desired_artifacts/` | Ảnh tham chiếu thiết kế |
| `data/vialectbench_finalized_{1,2,3,4,5}.json`, `data/vialectbench_finalized_full_mcqa_options.json` | Snapshot cũ; v6 là đầu vào hiện tại |
| `data/probe_dialects.jsonl`, `data/mcqa_full100.jsonl`, `data/mcqa_100_full_options.jsonl` | Probe pilot/bản trung gian, không được pipeline giữ lại sử dụng mặc định |
| `outputs/finalized_5_complete_six_intrinsic_analysis/` | Chỉ có 82 source đủ 6 dialect; không đại diện benchmark 400 source |
| `outputs/perplexity_analysis/`, `outputs/perplexity_unreviewed_visualization/` | Cache cũ dùng input khác, không phải intrinsic v6 |
| `scripts/quant_perplexity.py`, `scripts/visualize_perplexity.py` | Pipeline PPL cũ; bản finalized dùng `analyze_vialectbench_data.py` |
| `model_output/`, `docs/PA1.md`, `docs/PA2.md`, các PDF cũ trong `docs/`, `docs/sampling_and_benchmarking.md` | Kết quả pilot/proposal/hướng dẫn cũ; tài liệu bàn giao này thay cho hướng dẫn pilot |
| `papers/`, `AuthorKit27/`, `ArXiv_dialect/`, workspace LaTeX cũ, `main.tex`, `ui/` | Tài liệu tham khảo/template/bản thảo và UI ngoài pipeline rebuttal |

Các snapshot cũ đã commit vẫn truy cập được trên `main`. Bản sao trước khi dọn
của các file bị sửa, cùng diff và Git status ban đầu, nằm trong
`.local/rebuttal-cleanup/`; thư mục này bị ignore và có chứa thông tin cũ cần giữ riêng.

## Nối câu hỏi reviewer với dữ liệu/code

| Câu hỏi | Điểm bắt đầu | Giới hạn hiện tại |
| --- | --- | --- |
| Phân bố task/label, tiêu chí sampling, trigger density | 4 file mẫu nguồn, `selection_summary.csv`, lexicon, `src/select_samples.py`, `configs/datasets.yaml` | So sánh với toàn bộ source pool cần tải/xác định lại đúng pool và split; mẫu 100/task không thay thế pool gốc |
| Chất lượng benchmark, authenticity, semantic preservation | Annotation v6 và notebook quality audit | Audit cấu trúc/đối chiếu không xác minh được naturalness hay meaning preservation về ngôn ngữ |
| Table 1, agreement, fully passing subset | Cần export review từng reviewer và kết quả adjudication | V6 chỉ có metadata như `review_count`, `final_status`, `confidence`; không có đầy đủ rating Yes/Partial/No từng reviewer để tái tạo Table 1 hoặc tính agreement |
| QA exact match bên cạnh F1, harmful flips, case study | Raw prediction trong `outputs/direct_prompting/`, probe và EDA | Có thể tính lại từ cache; cần xuất thêm phân tích EM/QA theo yêu cầu reviewer |
| Rewrite density, nhóm biến đổi từ vựng, lỗi PNT2/PNT3 | Lexical records, alignment/review sheets và swap-NLL cache | NLL là tín hiệu của reference LM, không tự chứng minh nguyên nhân lỗi downstream |
| Kích thước model và robustness | 3 model Qwen2.5 và các output/bảng hiện có | Có thể so sánh mô tả; không suy ra tỉ lệ dialect trong pretraining hoặc kết luận causal |
| ViDia2Std/normalization baseline | `outputs/normalizer_full/README.md`, 10 file `outputs/normalized_direct_prompting_full/`, bảng full trong `paper_artifacts/direct_prompting/`; rescue riêng trong EDA | Baseline full dùng 2.400 rewrite × 10 model, giữ nguyên prompt/config gốc; phân tích cả gains, regressions và semantic drift |
| Phân tích tỉnh hoặc open-ended generation | Kiểm tra metadata annotation; có thể dùng runner/prompt làm điểm xuất phát | Không coi 6 dialect group là nhãn tỉnh; benchmark hiện tại không cung cấp thí nghiệm generation robustness |

Một điểm cần đối chiếu trước khi trả lời reviewer: paper ghi MCQA từ VIMQA, nhưng
`data/mcqa_100.jsonl` và `configs/datasets.yaml` ghi `uitnlp/vimmrc2.0`.
Việc dọn repo giữ nguyên dữ liệu/provenance này; nhóm tác giả cần xác nhận lại mô tả
nguồn trong paper. `main.tex` trong bản lưu trữ local là bản thảo cũ và không được coi là nguồn của
`submission.pdf`.

## Làm việc từ kết quả đã chạy

Chạy từ repo root:

```bash
python -m pip install -r requirements-analysis.txt
python scripts/check_rebuttal_bundle.py
python -m nbconvert --to notebook --execute notebooks/EDA.ipynb --output EDA.executed.ipynb --output-dir .local
python -m nbconvert --to notebook --execute notebooks/probe_dialects_quality_audit.ipynb --output audit.executed.ipynb --output-dir .local
python scripts/lexical_visualize.py
```

EDA ghi lại artifact trong `paper_artifacts/`, còn lexical visualization ghi hình
trong `outputs/lexical_substitution_analysis/figures/`. Review diff sau khi chạy.
EDA không tạo lại các hình PPL riêng đã có trong `paper_artifacts/dataset_analysis/`;
script intrinsic tạo bảng và SVG trong thư mục cache của nó. Không coi mọi artifact
hiện có là sản phẩm của riêng EDA.

Muốn tái tổng hợp intrinsic từ NLL đã cache, dùng `--reuse_scores` để tránh tải
model/chấm điểm lại; lệnh này vẫn ghi lại summary và hình trong output directory:

```bash
python scripts/analyze_vialectbench_data.py --complete_six_dialects_only --reuse_scores
```

Không chạy toàn bộ lexical prep mặc định chỉ để xem kết quả: các bước đầu có thể
tạo lại alignment hoặc review sheets. Ưu tiên đọc cache và giữ review sheets đã hoàn tất.
Nếu chủ động thay annotation/review, đọc `outputs/lexical_substitution_analysis/README.md`
và các tùy chọn `--steps` trước khi cập nhật.

`docs/rebuttal_manifest.json` ghi SHA-256 của 126 file dữ liệu/cache/artifact hiện
có, chuẩn hóa CRLF thành LF cho text để không lỗi do checkout trên Windows/Linux.
Manifest giúp phát hiện thiếu/thay đổi file khi bàn giao; vì được tạo sau inference,
nó không chứng minh chính xác input hoặc revision model của lần chạy cũ.
Sau khi nhóm đã review một snapshot kết quả mới, cập nhật có chủ ý:

```bash
python scripts/check_rebuttal_bundle.py --update-manifest
```

Đã kiểm tra trong lần dọn repo:

- Đủ 400 probe case, 100/task; đủ 2.400 annotation pair và 28.000 result row của 10 model.
- Chạy tuần tự toàn bộ 19 code cell EDA trên bản sao riêng, không gọi model/API.
- Chạy toàn bộ 11 code cell audit: 0 coverage issue, 0 format issue, 0 source issue, 0 probe-consistency issue.
- Conversion v6 qua `src.prepare_annotated_probe_cases` tạo đủ 400 case, không có duplicate sample/dialect.
- Probe tái tạo khớp hoàn toàn nội dung JSON với probe đang chia sẻ.
- Trong giai đoạn dọn repo ban đầu, không chạy training normalizer hoặc inference mới. Thí nghiệm normalization full bổ sung sau đó được lưu riêng ở `outputs/normalizer_full/`.

## Commit và chia sẻ

Giữ các thay đổi rebuttal trên nhánh `rebuttal/cleanup` và không sửa lịch sử
`main`. Sau khi review các file mới, commit rồi push nhánh riêng:

```bash
git status --short
git show --stat --oneline HEAD
git push -u origin rebuttal/cleanup
```

Sau khi nhánh đã commit, thành viên có thể tạo nhánh riêng từ `rebuttal/cleanup`.
Token Hugging Face từng ghi trực tiếp trong notebook đã được thay bằng biến môi
trường và notebook outputs đã được xóa. Nếu token đó vẫn còn hiệu lực, chủ token
cần thu hồi/đổi token vì bản cũ có thể vẫn nằm trong lịch sử `main`.
