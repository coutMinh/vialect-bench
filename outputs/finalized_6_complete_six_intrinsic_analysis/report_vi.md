# Báo cáo quantitative data analysis cho VialectBench finalized 6

## Mục tiêu

Phân tích này là intrinsic evaluation của dataset, không phải đánh giá năng lực của model. Reference causal language model chỉ được dùng như một scorer cố định để đo câu paraphrase phương ngữ lệch khỏi phân phối tiếng Việt chuẩn đến mức nào.

## Cách chạy

```bash
python scripts/analyze_vialectbench_data.py \
  --input data/vialectbench_finalized_6.json \
  --output_dir outputs/finalized_6_intrinsic_analysis \
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

- Tổng số dòng finalized: 2400
- Số sample_id duy nhất: 400
- Số sample_id có đủ 6 dialect: 400
- Chỉ giữ sample_id đủ 6 dialect khi chạy analysis: True
- Số cặp `(sample_id, target_dialect)` bị trùng: 0
- Dialect groups: PNB, PNN, PNT1, PNT2, PNT3, PNT4
- Tasks: MCQA, NLI, QA, SENT

### Distribution theo dialect

| Dialect | N |
| --- | --- |
| PNB | 400 |
| PNN | 400 |
| PNT1 | 400 |
| PNT2 | 400 |
| PNT3 | 400 |
| PNT4 | 400 |

### Distribution theo task

| Task | N |
| --- | --- |
| MCQA | 600 |
| NLI | 600 |
| QA | 600 |
| SENT | 600 |

## Kết quả intrinsic evaluation

Trên toàn bộ dataset, `delta_nll_mean = 0.482` với CI 95% [0.458, 0.507], `ratio_ppl_mean = 2.048`, và `length_ratio_mean = 0.990`.

Diễn giải: nếu `delta_nll_mean` dương, paraphrase phương ngữ nhìn chung khó dự đoán hơn với reference LM so với câu chuẩn cùng nội dung. Điều này không có nghĩa dữ liệu kém chất lượng; nó cho thấy dataset đưa vào tín hiệu dialectal variation có thể đo được.

### Theo dialect

| Dialect | N | Orig PPL | Dialect PPL | Delta NLL | CI low | CI high | PPL ratio | Length ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PNB | 400 | 93.178 | 82.909 | -0.104 | -0.129 | -0.080 | 0.926 | 0.995 |
| PNN | 400 | 93.178 | 136.919 | 0.316 | 0.277 | 0.355 | 1.493 | 0.984 |
| PNT1 | 400 | 93.178 | 166.437 | 0.468 | 0.415 | 0.520 | 1.905 | 0.997 |
| PNT2 | 400 | 93.178 | 257.017 | 0.795 | 0.738 | 0.853 | 2.726 | 0.991 |
| PNT3 | 400 | 93.178 | 263.194 | 0.925 | 0.860 | 0.991 | 3.334 | 0.989 |
| PNT4 | 400 | 93.178 | 158.199 | 0.495 | 0.447 | 0.545 | 1.903 | 0.986 |

### Theo task

| Task | N | Orig PPL | Dialect PPL | Delta NLL | CI low | CI high | PPL ratio | Length ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MCQA | 600 | 21.998 | 31.260 | 0.317 | 0.292 | 0.342 | 1.444 | 0.995 |
| NLI | 600 | 132.429 | 239.129 | 0.518 | 0.462 | 0.573 | 2.244 | 0.998 |
| QA | 600 | 47.171 | 103.105 | 0.642 | 0.590 | 0.695 | 2.385 | 0.985 |
| SENT | 600 | 171.115 | 336.290 | 0.453 | 0.400 | 0.506 | 2.118 | 0.982 |

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