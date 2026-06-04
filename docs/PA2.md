28/05
280 câu cho mỗi vùng miền, consider gộp các vùng lại với nhau (có thể nhỏ hơn)
Có 3 task chính mỗi task có 1 số lượng câu hỏi nhất định, và đã có dataset rồi → chọn một số lượng nhất định câu của mỗi dataset
Mỗi hướng tìm ra 1-2 phương pháp đại diện cho hướng đó rồi propose hướng mới trên method tốt nhất (VD test time adaptation thì có RAG, few shot prompting, LoRA) → không tốt thì không tốt trên vùng nào, nếu train trên tiếng miền trung thì nó có tốt hơn cho tiếng miền nam và miền bắc hay không
Research questions: train trên dialect miền Bắc thì performance các dialect còn lại thì sao, train trên tập mới thì performance tập cũ thì sao, add sample mới vào thì có tăng không.
Khi viết paper: List ra hypothesis, giả sử dùng LLMs thì performance khác nhau giữa các dialects group → dựa trên hypothesis thì design các experiments để tesk các hypothesis; làm thế này thì khi document và viết vào paper thì sẽ mạch lạc hơn rất nhiều.
xxx vào làm experiments luôn thì không make sense 
Từ hypothesis viết vào paper luôn → abstract, tùy vào quá trình chạy experiments có thể adjust paper song song với lúc chạy thí nghiệm
Tìm 1 aspect để phát triền từ paper https://arxiv.org/pdf/2603.10211  (safety,..), 
-đối với dataset thì tập dataset được annotate bởi máy với người thì những cái đó phải được verify như thế nào (check bởi a Luân)
-Các task nào paper Kiet et el chưa làm cho tiếng địa phương thì mình focus vào làm
-Có cách nào đo tiếng miền trung cho tập này là bao nhiêu, không cần làm miền Nam và miền Bắc. Trong bài này mình chỉ tập trung vào miền trung thôi --> không phải là tiếng phổ thông, stress test các hệ thông AI trên văn bản miền Trung
- Làm sao để đo được natural
(1) research gap / problem --> (2) proposal (algorithm/dataset) --> (3) validate whether the proposal tackles the problem 
29/05
Novelty
Paper ViDia2Std giới thiệu 1 dataset dialect to standard là những sentence được collect trên FB và annotate sang standard, experiments của họ chứng minh được rằng khi đã chuyển sang standard bằng model seq2seq pretrained on parallel corpus thì performance cải thiện (đương nhiên) → dataset tập trung vào việc xây dựng bộ chuyển đổi → nên chỉ đơn giản là collect các câu nói người dùng chat trên mxh rồi cần annotator chuyển sang standard  → thiếu insights rõ ràng về performance drop trên dialects nào, tại sao lại không bền vững (do sự thay đổi về noun, pronoun, hay từ đặc trưng,...), train trên 1 dialect ảnh hưởng đến các dialect còn lại không
→ Nói tóm lại bài của họ không có insights gì về model, dataset. Chỉ là đưa ra 1 bộ dataset song ngữ phục vụ cho training bộ chuyển đổi.
Paper mình làm benchmark các mô hình trên các task cụ thể có gold label (QA, NLI, …) cần những suy luận, kiến thức pretrained của mô hình can thiệp vào
Dataset của mình finegrained hơn (1 sample có 4-5 dialect variants) và vì thể nên mình có thể có nhiều hypothesis về ngôn ngữ các vùng miền.
Bài mình cần có insight sâu hơn về LLM hiện tại yếu trên dialect nào, một mô hình tốt nhất trên dialect nào …
Từ có improve model trên các task với dialect. Có thể thử train lại 1 model normalization của họ và thử trên các task của mình và có chứng minh được rằng normalization model oversimplify dialect word (xóa, thay bằng từ mang ít nghĩa hơn). Future work của họ cũng đề cập về 1 model normalization có thể nên giữ lại các dialectal quan trọng và limitation của họ là over-normalization
Comprare được một cái gì đó chứng minh được mình không cần dùng nhiều data mà mô hình vẫn hiểu được dialects → liệu phương pháp TTA của mình có tạo ra semantic comparable version standard để so với normalization không → từ đó chứng minh phương pháp của mình hiệu quả hơn về việc giữ lại được 

Chia vùng miền
Cần check lại cách chia vùng miền qua việc các vùng có thể share cùng vocab
Trung: Thanh Hóa, Nghệ Tĩnh, Bình Trị Thiện, Nam Trung Bộ → lý giải các vùng còn lại sử dụng phương ngữ của vùng kinh tế chính (lý do)
Task
Classification: NLI, sentiment.
Language understanding: MCQA, NER

Dataset construction
Cần design label strategy: Một câu standard → 1 người viết lại toàn bộ câu, hai người kiểm tra nếu cả 2 đòng ý thì giữ nếu 2 người không đồng ý thì viết lại còn nếu 1 k đồng ý thì tìm người thứ 3 (expert, đã có bằng đại học hoặc có qualification cao hơn); căn bản cần ít nhất 4 người cho 1 câu → 1 gr có 4 người
Làm 1 document để cho các bạn volunteer kí kết với mình, để đảm bảo không bị kiện về việc tiền bạc → consent agreement.
Tặng quà lưu niệm của seas cho các bạn (budget 200-300), giấy chứng nhận

Mục tiêu trước buổi họp tiếp theo ngày 4/6
Minh: Data seclection (difficulty, diversity) → mỗi task chọn từ các dataset 100 sample, phương pháp chọn sample trong 1 dataset: define 1 measurement để chứng minh các câu mình chọn maximum về vocab hoặc minimum về semantic
Trinh: Tìm hiểu quy trình làm dữ liệu liên quan đến tạo câu, khi tạo 1 câu mới thì làm sao để đánh giá quality control của 1 annotator trước khi thật sự gán (tham khảo paper related work). VD: 1 nhóm 4 người cùng nhau gán 1 số lượng câu challenge (10 câu) và cross check để check xem mỗi người có vượt qua minimum ngưỡng nào đó để bắt đầu gán. 

Plan before PA3:
Minh
List dataset cho từng task.
Chọn 100 samples/task bản thử nghiệm.
Định nghĩa diversity/difficulty.
Đề xuất model benchmark cho từng task.
Phân biệt model nào zero-shot, model nào pretrained/fine-tuned sẵn.
Chạy thử vài testcase fail trên ViDia2Std hoặc dialect examples.
Chuẩn bị 1 bảng hypothesis → experiment → expected insight.
Trinh
Annotation workflow.
Quality assurance protocol.
Annotator qualification/challenge set.
Consent/agreement form cho volunteer.
Ước lượng số câu cần paraphrase/review.
Số annotator có thể tham gia.
Kế hoạch phân công trên Label Studio hoặc web annotation.
