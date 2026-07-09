# Báo cáo quantitative data analysis cho VialectBench finalized 5

## Mục tiêu

Phân tích này là intrinsic evaluation của dataset, không phải đánh giá năng lực của model. Reference causal language model chỉ được dùng như một scorer cố định để đo câu paraphrase phương ngữ lệch khỏi phân phối tiếng Việt chuẩn đến mức nào.

## Cách chạy

```bash
python scripts/analyze_vialectbench_data.py \
  --input data/vialectbench_finalized_5.json \
  --output_dir outputs/finalized_5_intrinsic_analysis \
  --model Qwen/Qwen2.5-0.5B
```

Nếu chỉ muốn thống kê dataset và length ratio, chưa chạy perplexity:

```bash
python scripts/analyze_vialectbench_data.py \
  --input data/vialectbench_finalized_5.json \
  --skip_perplexity
```

## Các metric

- `orig_len`, `para_len`: số token thô theo khoảng trắng của câu gốc và câu phương ngữ.
- `length_ratio = para_len / orig_len`: kiểm tra paraphrase có bị dài/ngắn bất thường hay không. Giá trị gần 1 nghĩa là độ dài được giữ tương đối ổn định.
- `NLL`: negative log-likelihood trung bình theo token từ reference LM. NLL càng cao nghĩa là chuỗi càng ít quen thuộc với LM.
- `PPL = exp(NLL)`: perplexity, cách đọc trực quan hơn của NLL nhưng dễ bị scale lớn theo model/tokenizer.
- `delta_nll = NLL_para - NLL_original`: metric chính. Dương nghĩa là câu dialect gây distribution shift so với câu chuẩn; âm nghĩa là câu dialect quen thuộc hơn với reference LM trong cặp đó.
- `ratio_ppl = PPL_para / PPL_original`: tỉ lệ perplexity để đọc nhanh mức tăng/giảm, nhưng nên diễn giải phụ sau `delta_nll`.
- `95% bootstrap CI`: khoảng tin cậy bootstrap cho trung bình `delta_nll`.

Lưu ý quan trọng: với task NLI, baseline đúng là `hypothesis`, không phải `original_text`, vì `original_text` là premise còn `dialect_text` là hypothesis được viết lại.

## Tổng quan dataset

- Tổng số dòng finalized: 492
- Số sample_id duy nhất: 82
- Số sample_id có đủ 6 dialect: 82
- Chỉ giữ sample_id đủ 6 dialect khi chạy analysis: True
- Số cặp `(sample_id, target_dialect)` bị trùng: 0
- Dialect groups: PNB, PNN, PNT1, PNT2, PNT3, PNT4
- Tasks: MCQA, NLI, QA, SENT

### Distribution theo dialect

| Dialect | N |
| --- | --- |
| PNB | 82 |
| PNN | 82 |
| PNT1 | 82 |
| PNT2 | 82 |
| PNT3 | 82 |
| PNT4 | 82 |

### Distribution theo task

| Task | N |
| --- | --- |
| MCQA | 276 |
| NLI | 66 |
| QA | 66 |
| SENT | 84 |

## Kết quả intrinsic evaluation

Trên toàn bộ dataset, `delta_nll_mean = 0.398` với CI 95% [0.353, 0.444], `ratio_ppl_mean = 1.735`, và `length_ratio_mean = 0.988`.

Diễn giải: nếu `delta_nll_mean` dương, paraphrase phương ngữ nhìn chung khó dự đoán hơn với reference LM so với câu chuẩn cùng nội dung. Điều này không có nghĩa dữ liệu kém chất lượng; nó cho thấy dataset đưa vào tín hiệu dialectal variation có thể đo được.

### Theo dialect

| Dialect | N | Orig PPL | Dialect PPL | Delta NLL | CI low | CI high | PPL ratio | Length ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PNB | 82 | 65.570 | 55.027 | -0.098 | -0.153 | -0.048 | 0.929 | 1.000 |
| PNN | 82 | 65.570 | 100.552 | 0.250 | 0.164 | 0.337 | 1.399 | 0.978 |
| PNT1 | 82 | 65.570 | 123.630 | 0.517 | 0.385 | 0.651 | 2.047 | 0.988 |
| PNT2 | 82 | 65.570 | 155.375 | 0.732 | 0.630 | 0.843 | 2.385 | 0.989 |
| PNT3 | 82 | 65.570 | 147.648 | 0.668 | 0.582 | 0.762 | 2.151 | 0.985 |
| PNT4 | 82 | 65.570 | 106.928 | 0.321 | 0.235 | 0.410 | 1.501 | 0.987 |

### Theo task

| Task | N | Orig PPL | Dialect PPL | Delta NLL | CI low | CI high | PPL ratio | Length ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MCQA | 276 | 20.037 | 29.186 | 0.337 | 0.302 | 0.374 | 1.473 | 0.995 |
| NLI | 66 | 64.583 | 134.688 | 0.596 | 0.416 | 0.774 | 2.437 | 0.995 |
| QA | 66 | 72.087 | 154.525 | 0.524 | 0.359 | 0.684 | 2.101 | 0.989 |
| SENT | 84 | 210.834 | 349.615 | 0.344 | 0.206 | 0.479 | 1.758 | 0.960 |

## Insight chính

- `delta_nll` là bằng chứng định lượng rằng các paraphrase phương ngữ tạo distribution shift so với tiếng Việt chuẩn, trong khi vẫn giữ thiết kế paired theo cùng sample.
- `length_ratio` giúp kiểm tra shift này không chỉ đến từ việc câu dialect dài/ngắn bất thường. Nếu length ratio gần 1 thì gap perplexity đáng tin hơn như tín hiệu dialectal surface variation.
- So sánh theo dialect cho biết nhóm vùng nào tạo shift mạnh hơn/yếu hơn; so sánh theo task giúp tránh kết luận sai do MCQA, QA, NLI, SENT có độ dài và format rất khác nhau.

## Output files

- `dataset_metadata.json`: metadata và cấu hình chạy.
- `distribution_by_dialect.csv`, `distribution_by_task.csv`, `distribution_task_x_dialect.csv`: phân bố dữ liệu.
- `length_summary_overall.csv`, `length_summary_by_dialect.csv`, `length_summary_by_task.csv`: length ratio.
- `perplexity_pairwise_results.csv`: điểm NLL/PPL theo từng cặp câu.
- `perplexity_summary_overall.csv`, `perplexity_summary_by_dialect.csv`, `perplexity_summary_by_task.csv`: bảng summary chính.
- `fig_ppl_by_dialect.svg`, `fig_delta_nll_by_dialect.svg`, `fig_count_by_dialect.svg`, `fig_count_by_task.svg`: hình minh họa.