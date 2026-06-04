21/05

Minimal pair
Specific task:
Simple tasks: Q&A, emotions classification, translation.
Multilingual models, gemma for example. 
SAE (Sparse Autoencoder Feature) Feature Extraction
Chia nhóm giọng
Miền Bắc (Hà Nội)
Miền Trung (Huế - Nghệ An)
Miền Nam (TP-HCM) 
Định nghĩa rules paraphrase cho annotators, agreement
label studio. 
Đảm bảo performance drop 
Experiments design:
Text classification, sentiment classification, Q&A, NLI
Dataset: Test (trước mắt) ~100 câu cho mỗi task  (Dataset cần có nhiều domain cho 1 task)
Scope: Chọn một số dialect đặc trưng của mỗi vùng miền
Performance giữa các dialects vs standard → performance drop
Mitigation: làm thế nào để mô hình robust dưới dialects
Test time adaption (in context learning): retrieve sample relate to C when solving (A, B in dialectal form) - kiếm 5-6 pp cho hướng này apply với dataset của mình xem thử performance có cao hơn không
Adapter on top of LLMs (Efficent Finetuning): Có thể dò ra layer nào perform kém, train mỗi layer đó thôi. 
TODO: 
Trinh sẽ tìm những dataset tiếng việt cho những task trên và chia cho mỗi người
Minh sẽ trình bày về các phương pháp test-time adaptation cho các task NLI, classification, Q&A
Dataset sẽ được hoàn thiện trước 4/6:
Divide to
