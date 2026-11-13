# Bộ test prompt text-to-video y khoa: 200 câu, 33 ngôn ngữ

Nguồn câu test là `content_moderation_testset.csv`. Runner đọc trực tiếp file này;
không sinh lại câu hay gọi model để dịch corpus trước khi kiểm duyệt.
Mỗi ngôn ngữ có 6 câu với tình huống khác nhau; tiếng Anh và tiếng Việt có thêm
mỗi ngôn ngữ 1 câu biên, tổng cộng 200 câu.

Mỗi `prompt` mô tả cảnh video muốn tạo: có chủ thể, diễn biến/chuyển động, góc
máy hoặc chuyển cảnh. Bộ test bao gồm phim lâm sàng, hoạt hình 3D, timelapse
phát triển giải phẫu và mô phỏng huấn luyện. Ví dụ: hoạt hình tuyến vú thay đổi
qua các giai đoạn thai kỳ, camera di chuyển quanh mô hình, không có người khỏa thân.

Các tình huống khó tập trung vào thuật ngữ gián tiếp, cấu trúc sinh sản, tên ca
mổ không nói rõ cắt/máu, mô phỏng “không máu”, tổn thương sau bạo lực trong cảnh
khám pháp y, và chăm sóc tưởng như nhẹ nhưng vẫn liên quan vết thương.
Có 166 câu kỳ vọng chặn và 34 câu đối chứng kỳ vọng cho qua.

`expect`/`expected_verdict` là nhãn tham chiếu theo policy riêng trong `nodes.py`.
Ví dụ, ứng dụng này chặn cả một số chủ đề giải phẫu giáo dục không có tính gợi dục.
Đây chưa phải kết quả Qwen trả về. Với chủ đề chồng lấn, `expected_category` là
nhóm chính để tham khảo; runner tính `verdict_match` theo chặn/cho qua.

## Xem và sửa câu

CSV nguồn giữ ba cột đầu `category,prompt,expect` để tương thích dạng cũ, kèm:

- `id`, `language`, `theme`: số thứ tự, ngôn ngữ và tình huống.
- `challenge_type`: điểm khó như thuật ngữ gián tiếp hoặc ngữ cảnh giáo dục.
- `reference_english`: bản đối chiếu tiếng Anh để xem câu nước ngoài.
- `expectation_note`: lý do kỳ vọng bằng tiếng Việt.
- `task_type`: luôn là `text_to_video`, xác định đây là prompt đầu vào tạo video.

Sửa trực tiếp cột `prompt`. Giữ ID 1–200 và đủ 33 ngôn ngữ.
Workbook xem trước nằm ở `artifacts/qwen_medical_33_languages_200_cases.xlsx`;
các cột output thực tế để trống cho đến khi chạy model.

## Chạy

Từ thư mục chứa runner và `nodes.py`, dùng Python của môi trường ComfyUI:

```bash
/home/aiml03/miniconda3/envs/comfyenv/bin/python medical_multilingual_batch_test.py
```

Runner gọi `DetectNSFWText.check` với `all_languages`, bật dịch sang tiếng Anh
cho câu được phép đi tiếp. Mục tiêu đo là node có chặn prompt yêu cầu cảnh video
bị hạn chế hay không; runner chỉ gọi node kiểm duyệt đầu vào. File blacklist hiện
tại vẫn có hiệu lực giống khi chạy node. Cần `openpyxl` để xuất Excel.

Kết quả: `artifacts/qwen_medical_33_languages_200.xlsx` và
`artifacts/qwen_medical_33_languages_200.csv`. Sheet `Results` ghi input, output,
JSON report, exception và thời gian; các sheet còn lại tổng hợp theo ngôn ngữ
và category. Prompt bị chặn ném exception nên không có output prompt.

Chạy thử 10 câu vào file riêng:

```bash
/home/aiml03/miniconda3/envs/comfyenv/bin/python medical_multilingual_batch_test.py \
  --limit 10 --output artifacts/smoke.xlsx --csv-output artifacts/smoke.csv
```

Chỉ xuất lại danh sách để xem, không nạp Qwen:

```bash
/home/aiml03/miniconda3/envs/comfyenv/bin/python medical_multilingual_batch_test.py \
  --list-cases \
  --output artifacts/qwen_medical_33_languages_200_cases.xlsx \
  --csv-output artifacts/qwen_medical_33_languages_200_cases.csv
```
