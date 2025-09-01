# Cách sử dụng API mark-correct-answers

## Cách 1: Truyền JSON trực tiếp (không cần "correct_answers" wrapper)

```json
{
  "part1": {
    "1": "A",
    "2": "B",
    "3": "C",
    "4": "D",
    "5": "A",
    "6": "B",
    "7": "C",
    "9": "D",
    "10": "A",
    "11": "B",
    "12": "C",
    "13": "D",
    "14": "A",
    "15": "B",
    "16": "C",
    "17": "D",
    "18": "A",
    "19": "B",
    "20": "C",
    "21": "D",
    "22": "A",
    "23": "B",
    "24": "C",
    "25": "D",
    "26": "A",
    "27": "B",
    "28": "C",
    "29": "D",
    "30": "A",
    "31": "B",
    "32": "C",
    "33": "D",
    "34": "A",
    "35": "B",
    "36": "C",
    "37": "D",
    "38": "A",
    "39": "B",
    "40": "C"
  },
  "part2": {
    "1": "D,D,S,S",
    "2": "D,S,S,D",
    "3": "S,S,S,S",
    "4": "D,D,S,S",
    "5": "D,S,S,D",
    "6": "S,S,S,S",
    "7": "D,D,S,S",
    "8": "D,S,S,D"
  },
  "part3": {
    "1": "-3,2",
    "2": "2,1",
    "3": "1384",
    "4": "2025",
    "5": "20",
    "6": "200"
  }
}
```

## Cách 2: Truyền JSON với wrapper (cũng được hỗ trợ)

```json
{
  "correct_answers": {
    "part1": {
      "1": "A",
      "2": "B",
      "3": "C"
    },
    "part2": {
      "1": "D,D,S,S",
      "2": "D,S,S,D"
    },
    "part3": {
      "1": "-3,2",
      "2": "2,1"
    }
  }
}
```

## Cách gọi API bằng curl:

```bash
curl -X POST "http://localhost:8000/mark-correct-answers/" \
  -F "file=@your_image.jpg" \
  -F 'correct_answers={"part1":{"1":"A","2":"B","3":"C"},"part2":{"1":"D,D,S,S","2":"D,S,S,D"},"part3":{"1":"-3,2","2":"2,1"}}'
```

## Cách gọi API bằng Python:

```python
import requests
import json

url = "http://localhost:8000/mark-correct-answers/"

# Dữ liệu đáp án đúng
correct_answers = {
    "part1": {
        "1": "A",
        "2": "B",
        "3": "C"
    },
    "part2": {
        "1": "D,D,S,S",
        "2": "D,S,S,D"
    },
    "part3": {
        "1": "-3,2",
        "2": "2,1"
    }
}

# Gọi API
with open("your_image.jpg", "rb") as f:
    files = {"file": f}
    data = {"correct_answers": json.dumps(correct_answers)}

    response = requests.post(url, files=files, data=data)

    if response.status_code == 200:
        result = response.json()
        print("Success:", result)
    else:
        print("Error:", response.json())
```

## Giải thích format Part 2:

- Part 2 có cấu trúc đặc biệt: mỗi câu hỏi có 4 phần (a,b,c,d) và mỗi phần có 2 lựa chọn (D hoặc S)
- Format: "D,D,S,S" nghĩa là:
  - Phần a: chọn D
  - Phần b: chọn D
  - Phần c: chọn S
  - Phần d: chọn S

## Giải thích format Part 3:

- Part 3 dành cho điền số, có thể bao gồm dấu trừ (-) và dấu phẩy (,)
- Ví dụ:
  - "1": "-3,2" → Câu 1: dấu trừ (vị trí 1), số 3 (vị trí 2), dấu phẩy (vị trí 3), số 2 (vị trí 4)
  - "2": "2" → Câu 2: số 2 (vị trí 1)
  - "3": "1384" → Câu 3: số 1 (vị trí 1), số 3 (vị trí 2), số 8 (vị trí 3), số 4 (vị trí 4)

## Màu sắc đánh dấu:

- **Màu xanh lá (✓)**: Đáp án đúng và học sinh đã tô
- **Màu đỏ (✗)**: Đáp án đúng nhưng học sinh chưa tô

## Kết quả trả về:

```json
{
  "marked_image_path": "result/image_marked_answers.png",
  "summary": {
    "total_questions": { "part1": 5, "part2": 2, "part3": 2 },
    "marked_circles": { "part1": 5, "part2": 8, "part3": 3 },
    "correct_matches": {
      "part1": ["part1_1_a_236_702"],
      "part2": [],
      "part3": []
    },
    "incorrect_missing": {
      "part1": ["part1_2_b_280_702"],
      "part2": [],
      "part3": []
    },
    "unmarked_patterns": { "part1": [], "part2": [], "part3": [] }
  },
  "message": "Đánh dấu đáp án đúng thành công"
}
```
