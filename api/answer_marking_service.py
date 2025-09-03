import cv2
import numpy as np
import os
import json
import base64
import time
from typing import Dict, List, Any, Tuple, Union
from api.circle_detection_service import detect_circles


def find_exam_code_in_list(student_exam_code: str, exam_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Tìm mã đề của học sinh trong danh sách các mã đề

    Args:
        student_exam_code: Mã đề của học sinh
        exam_list: List các mã đề với đáp án

    Returns:
        Dict chứa thông tin mã đề và đáp án tương ứng

    Raises:
        ValueError: Nếu không tìm thấy mã đề phù hợp
    """
    for exam in exam_list:
        exam_code = exam.get("code", "")
        if exam_code == student_exam_code:
            return exam

    # Không tìm thấy mã đề phù hợp
    available_codes = [exam.get("code", "") for exam in exam_list]
    raise ValueError(f"Không tìm thấy mã đề '{student_exam_code}' trong danh sách. Các mã đề có sẵn: {available_codes}")


def parse_correct_answers_new_format(correct_answers: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    Parse correct answers từ array format mới thành danh sách các circle labels cần đánh dấu

    Args:
        correct_answers: List chứa các section với đáp án đúng

    Returns:
        Dict với key là part và value là list các circle labels cần đánh dấu
    """
    marked_circles = {
        "part1": [],
        "part2": [],
        "part3": []
    }

    for section in correct_answers:
        section_type = section.get("sectionType", "")
        questions = section.get("questions", [])

        if section_type == "MULTIPLE_CHOICE":
            # Xử lý Part 1
            for question in questions:
                question_num = str(question.get("questionNumber", ""))
                answer = question.get("answer", "").lower()
                # Format: part1_{question_num}_{answer}_x_y
                marked_circles["part1"].append(f"part1_{question_num}_{answer}")

        elif section_type == "TRUE_FALSE":
            # Xử lý Part 2
            for question in questions:
                question_num = str(question.get("questionNumber", ""))
                answer_dict = question.get("answer", {})

                for sub_part, answer in answer_dict.items():
                    # Chuyển đổi "Đ" thành "D" và "S" giữ nguyên
                    if answer == "Đ":
                        answer = "D"
                    elif answer == "S":
                        answer = "S"
                    # Format: part2_{question_num}_{sub_part}_{answer}_x_y
                    marked_circles["part2"].append(f"part2_{question_num}_{sub_part}_{answer}")

        elif section_type == "ESSAY_CODE":
            # Xử lý Part 3
            for question in questions:
                question_num = str(question.get("questionNumber", ""))
                answer_str = str(question.get("answer", ""))
                # Parse answer string để tìm các ký tự và vị trí của chúng
                patterns = parse_part3_answer(question_num, answer_str)
                marked_circles["part3"].extend(patterns)

    return marked_circles


def parse_correct_answers(correct_answers) -> Dict[str, List[str]]:
    """
    Parse correct answers từ JSON format thành danh sách các circle labels cần đánh dấu
    Hỗ trợ cả format cũ (Dict) và format mới (List)

    Args:
        correct_answers: Dict hoặc List chứa đáp án đúng

    Returns:
        Dict với key là part và value là list các circle labels cần đánh dấu
    """
    # Kiểm tra format mới (List) hay cũ (Dict)
    if isinstance(correct_answers, list):
        return parse_correct_answers_new_format(correct_answers)

    # Format cũ (Dict)
    marked_circles = {
        "part1": [],
        "part2": [],
        "part3": []
    }

    # Xử lý Part 1
    if "part1" in correct_answers:
        for question_num, answer in correct_answers["part1"].items():
            answer_lower = answer.lower()
            # Format: part1_{question_num}_{answer}_x_y
            marked_circles["part1"].append(f"part1_{question_num}_{answer_lower}")

    # Xử lý Part 2
    if "part2" in correct_answers:
        for question_num, answers_data in correct_answers["part2"].items():
            # Handle both formats:
            # Format 1: "D,D,S,S" (string)
            # Format 2: {"a": "D", "b": "S", "c": "D", "d": "S"} (dict)

            if isinstance(answers_data, str):
                # Format 1: answers_data is string like "D,D,S,S"
                answers_list = answers_data.split(",")
                sub_parts = ["a", "b", "c", "d"]

                for i, answer in enumerate(answers_list):
                    if i < len(sub_parts):
                        sub_part = sub_parts[i]
                        answer = answer.strip()
                        # Format: part2_{question_num}_{sub_part}_{answer}_x_y
                        marked_circles["part2"].append(f"part2_{question_num}_{sub_part}_{answer}")

            elif isinstance(answers_data, dict):
                # Format 2: answers_data is dict like {"a": "D", "b": "S", "c": "D", "d": "S"}
                for sub_part, answer in answers_data.items():
                    answer = answer.strip()
                    # Format: part2_{question_num}_{sub_part}_{answer}_x_y
                    marked_circles["part2"].append(f"part2_{question_num}_{sub_part}_{answer}")

    # Xử lý Part 3
    if "part3" in correct_answers:
        for question_num, answer_str in correct_answers["part3"].items():
            # Parse answer string để tìm các ký tự và vị trí của chúng
            patterns = parse_part3_answer(question_num, answer_str)
            marked_circles["part3"].extend(patterns)

    return marked_circles


def parse_part3_answer(question_num: str, answer_str: str) -> List[str]:
    """
    Parse đáp án Part 3 thành các pattern circles cần đánh dấu
    
    Args:
        question_num: Số câu hỏi (ví dụ: "1", "2")
        answer_str: Chuỗi đáp án (ví dụ: "-3,2", "2", "1384")
    
    Returns:
        List các pattern cần tìm
    """
    patterns = []
    position = 1  # Vị trí bắt đầu từ 1
    
    for char in answer_str:
        if char == '-':
            # Dấu trừ
            pattern = f"part3_{question_num}_minus_{position}"
            patterns.append(pattern)
        elif char == ',':
            # Dấu phẩy
            pattern = f"part3_{question_num}_comma_{position}"
            patterns.append(pattern)
        elif char.isdigit():
            # Số
            pattern = f"part3_{question_num}_{char}_{position}"
            patterns.append(pattern)
        # Bỏ qua các ký tự khác (nếu có)
        
        position += 1
    
    return patterns


def convert_to_new_format(student_answers: Dict[str, Any], student_id: str, exam_code: str, image_path: str) -> Dict[str, Any]:
    """
    Chuyển đổi format đáp án từ format cũ sang format mới theo yêu cầu

    Args:
        student_answers: Dict đáp án theo format cũ
        student_id: Mã học sinh
        exam_code: Mã đề thi
        base64_image: Ảnh base64

    Returns:
        Dict theo format mới
    """
    student_answer_json = []

    # Section 1: MULTIPLE_CHOICE (Part 1) - Always include
    questions_part1 = []
    if "part1" in student_answers and student_answers["part1"]:
        for question_num, answer in student_answers["part1"].items():
            try:
                # Extract only numeric part from question_num
                numeric_part = ''.join(filter(str.isdigit, str(question_num)))
                if numeric_part:
                    questions_part1.append({
                        "questionNumber": int(numeric_part),
                        "answer": answer.upper()
                    })
            except (ValueError, TypeError) as e:
                print(f"Warning: Skipping invalid question_num '{question_num}' in part1: {e}")
                continue

        # Sort by question number
        questions_part1.sort(key=lambda x: x["questionNumber"])

    student_answer_json.append({
        "sectionOrder": 1,
        "sectionType": "MULTIPLE_CHOICE",
        "questions": questions_part1
    })

    # Section 2: TRUE_FALSE (Part 2) - Always include
    questions_part2 = []
    if "part2" in student_answers and student_answers["part2"]:
        for question_num, sub_answers in student_answers["part2"].items():
            try:
                # Extract only numeric part from question_num
                numeric_part = ''.join(filter(str.isdigit, str(question_num)))
                if not numeric_part:
                    continue

                if isinstance(sub_answers, dict):
                    # Format: {"a": "D", "b": "S", "c": "D", "d": "S"}
                    converted_answers = {}
                    for sub_part, answer in sub_answers.items():
                        converted_answers[sub_part] = "D" if answer.upper() == "D" else "S"

                    questions_part2.append({
                        "questionNumber": int(numeric_part),
                        "answer": converted_answers
                    })

                elif isinstance(sub_answers, str):
                    # Format: "D,D,S,S"
                    answers_list = sub_answers.split(",")
                    sub_parts = ["a", "b", "c", "d"]

                    converted_answers = {}
                    for i, answer in enumerate(answers_list):
                        if i < len(sub_parts):
                            sub_part = sub_parts[i]
                            converted_answers[sub_part] = "D" if answer.strip().upper() == "D" else "S"

                    questions_part2.append({
                        "questionNumber": int(numeric_part),
                        "answer": converted_answers
                    })
            except (ValueError, TypeError) as e:
                print(f"Warning: Skipping invalid question_num '{question_num}' in part2: {e}")
                continue

        # Sort by question number
        questions_part2.sort(key=lambda x: x["questionNumber"])

    student_answer_json.append({
        "sectionOrder": 2,
        "sectionType": "TRUE_FALSE",
        "questions": questions_part2
    })

    # Section 3: ESSAY_CODE (Part 3) - Always include
    questions_part3 = []
    if "part3" in student_answers and student_answers["part3"]:
        for question_num, answer in student_answers["part3"].items():
            try:
                # Extract only numeric part from question_num (e.g., "1a" -> "1")
                numeric_part = ''.join(filter(str.isdigit, str(question_num)))
                if numeric_part:
                    questions_part3.append({
                        "questionNumber": int(numeric_part),
                        "answer": str(answer)
                    })
            except (ValueError, TypeError) as e:
                print(f"Warning: Skipping invalid question_num '{question_num}' in part3: {e}")
                continue

        # Sort by question number
        questions_part3.sort(key=lambda x: x["questionNumber"])

    student_answer_json.append({
        "sectionOrder": 3,
        "sectionType": "ESSAY_CODE",
        "questions": questions_part3
    })

    return {
        "student_code": student_id,
        "exam_code": exam_code,
        "image_path": image_path,  # Trả về đường dẫn ảnh thay vì base64
        "student_answer_json": student_answer_json
    }


def image_to_base64(image_array: np.ndarray) -> str:
    """
    Chuyển đổi ảnh numpy array thành base64 string

    Args:
        image_array: NumPy array của ảnh

    Returns:
        Base64 string của ảnh
    """
    # Encode ảnh thành PNG
    _, buffer = cv2.imencode('.png', image_array)
    # Chuyển thành base64
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"


def create_student_answers_debug_image(image: np.ndarray, student_answers: List[str], all_circles: List[str]) -> np.ndarray:
    """
    Tạo ảnh debug hiển thị đáp án của học sinh

    Args:
        image: Ảnh gốc
        student_answers: List các circle labels mà học sinh đã tô
        all_circles: List tất cả circle labels có thể

    Returns:
        NumPy array của ảnh debug
    """
    debug_image = image.copy()

    # Đánh dấu tất cả các ô có thể (màu xám nhạt)
    for circle_label in all_circles:
        x, y = extract_coordinates_from_label(circle_label)
        if x > 0 and y > 0:
            cv2.circle(debug_image, (x, y), 8, (200, 200, 200), 1)  # Màu xám nhạt

    # Đánh dấu các ô học sinh đã tô (màu xanh dương đậm)
    for circle_label in student_answers:
        x, y = extract_coordinates_from_label(circle_label)
        if x > 0 and y > 0:
            cv2.circle(debug_image, (x, y), 10, (255, 0, 0), 3)  # Màu xanh dương đậm

            # Thêm text để hiển thị thông tin chi tiết
            parts = circle_label.split("_")
            if len(parts) >= 3:
                if parts[0] == "part1":
                    # part1_1_a_x_y -> "1A"
                    text = f"{parts[1]}{parts[2].upper()}"
                elif parts[0] == "part2":
                    # part2_1_a_D_x_y -> "1a:D"
                    text = f"{parts[1]}{parts[2]}:{parts[3]}"
                elif parts[0] == "part3":
                    # part3_1_2_1_x_y -> "1:2@1"
                    symbol = parts[2]
                    if symbol == "minus":
                        symbol = "-"
                    elif symbol == "comma":
                        symbol = ","
                    text = f"{parts[1]}:{symbol}@{parts[3]}"
                elif parts[0] == "id":
                    # id_student_1_2_x_y -> "ID:1@2"
                    text = f"ID:{parts[2]}@{parts[3]}"
                elif parts[0] == "exam":
                    # exam_code_1_2_x_y -> "EX:1@2"
                    text = f"EX:{parts[2]}@{parts[3]}"
                else:
                    text = f"{parts[0][:2]}"

                # Vẽ text với background
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.4
                thickness = 1
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

                # Vẽ background cho text
                cv2.rectangle(debug_image,
                            (x - text_size[0]//2 - 2, y - 20 - text_size[1] - 2),
                            (x + text_size[0]//2 + 2, y - 20 + 2),
                            (255, 255, 255), -1)

                # Vẽ text
                cv2.putText(debug_image, text,
                          (x - text_size[0]//2, y - 20),
                          font, font_scale, (0, 0, 0), thickness)

    # Thêm legend
    legend_y = 30
    cv2.putText(debug_image, "Student Answers Debug", (10, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    legend_y += 25
    cv2.circle(debug_image, (20, legend_y), 8, (200, 200, 200), 1)
    cv2.putText(debug_image, "All possible circles", (40, legend_y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

    legend_y += 20
    cv2.circle(debug_image, (20, legend_y), 10, (255, 0, 0), 3)
    cv2.putText(debug_image, f"Student selected ({len(student_answers)} circles)", (40, legend_y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    return debug_image


def create_detailed_debug_image(image: np.ndarray, student_answers: List[str], all_circles: List[str],
                               multiple_part1: Dict[str, List[str]], multiple_part2: Dict[str, List[str]],
                               multiple_part3: Dict[str, List[str]], missing_answers: Dict[str, List[str]]) -> np.ndarray:
    """
    Tạo ảnh debug chi tiết với tất cả các vấn đề được đánh dấu

    Args:
        image: Ảnh gốc
        student_answers: List các circle labels mà học sinh đã tô
        all_circles: List tất cả circle labels có thể
        multiple_part1: Dict các câu Part 1 có nhiều đáp án
        multiple_part2: Dict các câu Part 2 có nhiều đáp án
        multiple_part3: Dict các vấn đề Part 3
        missing_answers: Dict các đáp án bị thiếu

    Returns:
        NumPy array của ảnh debug chi tiết
    """
    debug_image = image.copy()

    # Set để theo dõi các circles đã được đánh dấu
    marked_circles = set()

    # 1. Đánh dấu màu vàng cho các trường hợp nhiều đáp án
    # Part 1: Nhiều hơn 1 đáp án trong cùng câu
    for _, duplicate_answers in multiple_part1.items():
        for circle_label in duplicate_answers:
            x, y = extract_coordinates_from_label(circle_label)
            if x > 0 and y > 0:
                cv2.circle(debug_image, (x, y), 12, (0, 255, 255), 3)  # Yellow
                cv2.putText(debug_image, "MULTI", (x-15, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
                marked_circles.add(circle_label)

    # Part 2: Cả D và S trong cùng sub-question
    for _, duplicate_answers in multiple_part2.items():
        for circle_label in duplicate_answers:
            x, y = extract_coordinates_from_label(circle_label)
            if x > 0 and y > 0:
                cv2.circle(debug_image, (x, y), 12, (0, 255, 255), 3)  # Yellow
                cv2.putText(debug_image, "D&S", (x-10, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
                marked_circles.add(circle_label)

    # Part 3: Ô thừa và nhiều ký tự cùng vị trí
    for circle_label in multiple_part3.get("extra_answers", []):
        x, y = extract_coordinates_from_label(circle_label)
        if x > 0 and y > 0:
            cv2.circle(debug_image, (x, y), 12, (0, 255, 255), 3)  # Yellow
            cv2.putText(debug_image, "EXTRA", (x-15, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
            marked_circles.add(circle_label)

    for _, duplicate_answers in multiple_part3.get("multiple_at_position", {}).items():
        for circle_label in duplicate_answers:
            x, y = extract_coordinates_from_label(circle_label)
            if x > 0 and y > 0:
                cv2.circle(debug_image, (x, y), 12, (0, 255, 255), 3)  # Yellow
                cv2.putText(debug_image, "DUP", (x-10, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
                marked_circles.add(circle_label)

    # 2. Đánh dấu màu đỏ cho đáp án đúng bị thiếu
    for _, missing_circles in missing_answers.items():
        for circle_label in missing_circles:
            if circle_label not in marked_circles:
                x, y = extract_coordinates_from_label(circle_label)
                if x > 0 and y > 0:
                    cv2.circle(debug_image, (x, y), 12, (0, 0, 255), 3)  # Red
                    cv2.putText(debug_image, "MISS", (x-15, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
                    marked_circles.add(circle_label)

    # 3. Đánh dấu màu xanh cho đáp án đúng học sinh đã chọn
    for circle_label in student_answers:
        if circle_label not in marked_circles:
            x, y = extract_coordinates_from_label(circle_label)
            if x > 0 and y > 0:
                cv2.circle(debug_image, (x, y), 10, (0, 255, 0), 2)  # Green
                marked_circles.add(circle_label)

    # 4. Đánh dấu các ô còn lại (màu xám nhạt)
    for circle_label in all_circles:
        if circle_label not in marked_circles:
            x, y = extract_coordinates_from_label(circle_label)
            if x > 0 and y > 0:
                cv2.circle(debug_image, (x, y), 6, (150, 150, 150), 1)  # Light gray

    # Thêm legend chi tiết
    legend_x, legend_y = 10, 30
    cv2.putText(debug_image, "Detailed Debug Legend", (legend_x, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    legend_items = [
        ((0, 255, 255), "Multiple answers (Yellow)"),
        ((0, 0, 255), "Missing correct answers (Red)"),
        ((0, 255, 0), "Correct student answers (Green)"),
        ((150, 150, 150), "Other circles (Gray)")
    ]

    for i, (color, text) in enumerate(legend_items):
        y_pos = legend_y + 25 + (i * 20)
        cv2.circle(debug_image, (legend_x + 10, y_pos), 8, color, -1 if color != (150, 150, 150) else 1)
        cv2.putText(debug_image, text, (legend_x + 30, y_pos + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return debug_image


def parse_student_id_from_answers(student_answers: List[str]) -> str:
    """
    Parse student ID từ các circle labels của ID student
    
    Args:
        student_answers: List các circle labels mà học sinh đã tô
        
    Returns:
        String student ID (ví dụ: "123456")
    """
    id_digits = [""] * 6  # 6 vị trí cho student ID
    
    for answer in student_answers:
        if answer.startswith("id_student_"):
            parts = answer.split("_")
            if len(parts) >= 5:
                # Format: id_student_digit_position_x_y
                digit = parts[2]     # số 0-9
                position = int(parts[3]) - 1  # vị trí 1-6, chuyển về 0-5
                
                if 0 <= position < 6:
                    id_digits[position] = digit
    
    # Ghép lại student ID từ trái sang phải
    student_id = "".join(id_digits).rstrip()
    return student_id


def parse_exam_code_from_answers(student_answers: List[str]) -> str:
    """
    Parse exam code từ các circle labels của exam code
    
    Args:
        student_answers: List các circle labels mà học sinh đã tô
        
    Returns:
        String exam code (ví dụ: "123")
    """
    exam_code_digits = [""] * 3  # Chỉ 3 vị trí cho exam code
    
    for answer in student_answers:
        if answer.startswith("exam_code_"):
            parts = answer.split("_")
            if len(parts) >= 5:
                # Format: exam_code_digit_position_x_y
                digit = parts[2]     # số 0-9
                position = int(parts[3]) - 1  # vị trí 1-3, chuyển về 0-2
                
                if 0 <= position < 3:
                    exam_code_digits[position] = digit
    
    # Ghép lại exam code từ trái sang phải, loại bỏ các vị trí trống cuối
    exam_code = "".join(exam_code_digits).rstrip()
    return exam_code


def parse_student_answers(student_answers: List[str]) -> Dict[str, Dict[str, str]]:
    """
    Parse đáp án học sinh từ circle labels thành format mong muốn
    
    Args:
        student_answers: List các circle labels mà học sinh đã tô
        
    Returns:
        Dict chứa đáp án học sinh theo format: {"part1": {"1": "A"}, "part2": {"1a": "D"}, "part3": {"1": "-3,2"}}
    """
    result = {
        "part1": {},
        "part2": {},
        "part3": {}
    }
    
    # Tạo dict tạm để nhóm part3 theo câu hỏi và vị trí
    part3_temp = {}
    
    for answer in student_answers:
        parts = answer.split("_")
        if len(parts) < 4:
            continue
            
        part = parts[0]  # part1, part2, part3
        question_num = parts[1]  # số câu hỏi
        
        if part == "part1":
            # Format: part1_1_a_x_y -> "1": "A"
            if len(parts) >= 5:
                answer_choice = parts[2].upper()
                result["part1"][question_num] = answer_choice
                
        elif part == "part2":
            # Format: part2_1_a_D_x_y -> "1a": "D"
            if len(parts) >= 6:
                sub_part = parts[2]  # a, b, c, d
                choice = parts[3]    # D hoặc S
                key = f"{question_num}{sub_part}"
                result["part2"][key] = choice
                
        elif part == "part3":
            # Format: part3_1_3_2_x_y -> cần ghép lại thành số
            if len(parts) >= 6:
                symbol = parts[2]    # minus, comma, hoặc số
                position = parts[3]  # vị trí trong câu trả lời
                
                if question_num not in part3_temp:
                    part3_temp[question_num] = {}
                
                # Chuyển đổi symbol thành ký tự thực tế
                if symbol == "minus":
                    char = "-"
                elif symbol == "comma":
                    char = ","
                elif symbol.isdigit():
                    char = symbol
                else:
                    continue
                
                part3_temp[question_num][int(position)] = char
    
    # Ghép lại part3 theo thứ tự vị trí
    for question_num, positions in part3_temp.items():
        sorted_positions = sorted(positions.keys())
        answer_str = "".join(positions[pos] for pos in sorted_positions)
        result["part3"][question_num] = answer_str
    
    return result


def find_matching_circles(all_circles: List[str], target_patterns: List[str]) -> List[str]:
    """
    Tìm các circle labels từ all_circles khớp với target_patterns
    
    Args:
        all_circles: List tất cả circle labels từ detect_circles
        target_patterns: List các pattern cần tìm (không có tọa độ x,y)
        
    Returns:
        List các circle labels đầy đủ (có tọa độ) khớp với patterns
    """
    matching_circles = []
    
    for pattern in target_patterns:
        for circle_label in all_circles:
            # Kiểm tra nếu circle_label bắt đầu với pattern
            if circle_label.startswith(pattern + "_"):
                matching_circles.append(circle_label)
                break  # Chỉ lấy circle đầu tiên khớp
    
    return matching_circles


def extract_coordinates_from_label(circle_label: str) -> Tuple[int, int]:
    """
    Trích xuất tọa độ x, y từ circle label

    Args:
        circle_label: Chuỗi dạng "part1_1_a_236_702"

    Returns:
        Tuple (x, y) tọa độ
    """
    parts = circle_label.split("_")
    if len(parts) >= 2:
        try:
            x = int(parts[-2])
            y = int(parts[-1])
            return x, y
        except ValueError:
            pass
    return 0, 0


def check_multiple_answers_part1(student_answers: List[str]) -> Dict[str, List[str]]:
    """
    Kiểm tra các câu trong Part 1 có nhiều hơn 1 đáp án được chọn

    Args:
        student_answers: List các circle labels mà học sinh đã tô

    Returns:
        Dict với key là question_num và value là list các circle labels bị trùng
    """
    part1_answers = {}
    multiple_answers = {}

    for answer in student_answers:
        if answer.startswith("part1_"):
            parts = answer.split("_")
            if len(parts) >= 5:
                question_num = parts[1]

                if question_num not in part1_answers:
                    part1_answers[question_num] = []
                part1_answers[question_num].append(answer)

    # Tìm các câu có nhiều hơn 1 đáp án
    for question_num, answers in part1_answers.items():
        if len(answers) > 1:
            multiple_answers[question_num] = answers

    return multiple_answers


def check_multiple_answers_part2(student_answers: List[str]) -> Dict[str, List[str]]:
    """
    Kiểm tra các câu trong Part 2 có cả D và S trong cùng 1 sub-question

    Args:
        student_answers: List các circle labels mà học sinh đã tô

    Returns:
        Dict với key là "question_num_sub_part" và value là list các circle labels bị trùng
    """
    part2_answers = {}
    multiple_answers = {}

    for answer in student_answers:
        if answer.startswith("part2_"):
            parts = answer.split("_")
            if len(parts) >= 6:
                question_num = parts[1]
                sub_part = parts[2]
                choice = parts[3]  # D hoặc S

                key = f"{question_num}_{sub_part}"
                if key not in part2_answers:
                    part2_answers[key] = []
                part2_answers[key].append(answer)

    # Tìm các sub-question có cả D và S
    for key, answers in part2_answers.items():
        if len(answers) > 1:
            # Kiểm tra xem có cả D và S không
            has_d = any("_D_" in answer for answer in answers)
            has_s = any("_S_" in answer for answer in answers)
            if has_d and has_s:
                multiple_answers[key] = answers

    return multiple_answers


def check_multiple_answers_part3_new_format(student_answers: List[str], correct_answers: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    Kiểm tra Part 3 có các vấn đề (format mới):
    1. Học sinh đánh ô thừa (không có trong đáp án)
    2. Cùng 1 vị trí có nhiều ký tự được chọn

    Args:
        student_answers: List các circle labels mà học sinh đã tô
        correct_answers: List chứa các section với đáp án đúng

    Returns:
        Dict với key là loại lỗi và value là list các circle labels bị lỗi
    """
    part3_student = {}
    part3_correct_positions = {}
    issues = {
        "extra_answers": [],  # Ô thừa
        "multiple_at_position": {}  # Nhiều ký tự cùng vị trí
    }

    # Parse đáp án đúng để biết các vị trí hợp lệ
    for section in correct_answers:
        if section.get("sectionType") == "ESSAY_CODE":
            questions = section.get("questions", [])
            for question in questions:
                question_num = str(question.get("questionNumber", ""))
                answer_str = str(question.get("answer", ""))
                part3_correct_positions[question_num] = len(answer_str)

    # Nhóm đáp án học sinh theo câu hỏi và vị trí
    for answer in student_answers:
        if answer.startswith("part3_"):
            parts = answer.split("_")
            if len(parts) >= 6:
                question_num = parts[1]
                position = parts[3]

                if question_num not in part3_student:
                    part3_student[question_num] = {}
                if position not in part3_student[question_num]:
                    part3_student[question_num][position] = []

                part3_student[question_num][position].append(answer)

    # Kiểm tra các vấn đề
    for question_num, positions in part3_student.items():
        max_valid_position = part3_correct_positions.get(question_num, 0)

        for position, answers in positions.items():
            position_int = int(position)

            # Kiểm tra ô thừa
            if position_int > max_valid_position:
                issues["extra_answers"].extend(answers)

            # Kiểm tra nhiều ký tự cùng vị trí
            if len(answers) > 1:
                key = f"{question_num}_{position}"
                issues["multiple_at_position"][key] = answers

    return issues


def check_multiple_answers_part3(student_answers: List[str], correct_answers) -> Dict[str, List[str]]:
    """
    Kiểm tra Part 3 có các vấn đề:
    1. Học sinh đánh ô thừa (không có trong đáp án)
    2. Cùng 1 vị trí có nhiều ký tự được chọn
    Hỗ trợ cả format cũ (Dict) và format mới (List)

    Args:
        student_answers: List các circle labels mà học sinh đã tô
        correct_answers: Dict hoặc List chứa đáp án đúng

    Returns:
        Dict với key là loại lỗi và value là list các circle labels bị lỗi
    """
    # Kiểm tra format mới (List) hay cũ (Dict)
    if isinstance(correct_answers, list):
        return check_multiple_answers_part3_new_format(student_answers, correct_answers)

    # Format cũ (Dict)
    part3_student = {}
    part3_correct_positions = {}
    issues = {
        "extra_answers": [],  # Ô thừa
        "multiple_at_position": {}  # Nhiều ký tự cùng vị trí
    }

    # Parse đáp án đúng để biết các vị trí hợp lệ
    if "part3" in correct_answers:
        for question_num, answer_str in correct_answers["part3"].items():
            part3_correct_positions[question_num] = len(answer_str)

    # Nhóm đáp án học sinh theo câu hỏi và vị trí
    for answer in student_answers:
        if answer.startswith("part3_"):
            parts = answer.split("_")
            if len(parts) >= 6:
                question_num = parts[1]
                position = parts[3]

                if question_num not in part3_student:
                    part3_student[question_num] = {}
                if position not in part3_student[question_num]:
                    part3_student[question_num][position] = []

                part3_student[question_num][position].append(answer)

    # Kiểm tra các vấn đề
    for question_num, positions in part3_student.items():
        max_valid_position = part3_correct_positions.get(question_num, 0)

        for position, answers in positions.items():
            position_int = int(position)

            # Kiểm tra ô thừa
            if position_int > max_valid_position:
                issues["extra_answers"].extend(answers)

            # Kiểm tra nhiều ký tự cùng vị trí
            if len(answers) > 1:
                key = f"{question_num}_{position}"
                issues["multiple_at_position"][key] = answers

    return issues


def check_missing_answers_new_format(student_answers: List[str], correct_answers: List[Dict[str, Any]], all_circles: List[str]) -> Dict[str, List[str]]:
    """
    Kiểm tra các câu không có đáp án nào được chọn (format mới)

    Args:
        student_answers: List các circle labels mà học sinh đã tô
        correct_answers: List chứa các section với đáp án đúng
        all_circles: List tất cả circle labels có thể

    Returns:
        Dict với key là part và value là list các đáp án đúng bị thiếu
    """
    missing_answers = {
        "part1": [],
        "part2": [],
        "part3": []
    }

    for section in correct_answers:
        section_type = section.get("sectionType", "")
        questions = section.get("questions", [])

        if section_type == "MULTIPLE_CHOICE":
            # Part 1: Kiểm tra từng câu
            for question in questions:
                question_num = str(question.get("questionNumber", ""))
                answer = question.get("answer", "").lower()

                # Kiểm tra xem câu này có đáp án nào được chọn không
                has_answer = any(answer_label.startswith(f"part1_{question_num}_") for answer_label in student_answers)
                if not has_answer:
                    # Tìm đáp án đúng cho câu này
                    correct_pattern = f"part1_{question_num}_{answer}"
                    matching_circles = find_matching_circles(all_circles, [correct_pattern])
                    missing_answers["part1"].extend(matching_circles)

        elif section_type == "TRUE_FALSE":
            # Part 2: Kiểm tra từng sub-question
            for question in questions:
                question_num = str(question.get("questionNumber", ""))
                answer_dict = question.get("answer", {})

                for sub_part, answer in answer_dict.items():
                    # Chuyển đổi "Đ" thành "D"
                    if answer == "Đ":
                        answer = "D"
                    elif answer == "S":
                        answer = "S"

                    # Kiểm tra sub-question này có đáp án không
                    has_answer = any(answer_label.startswith(f"part2_{question_num}_{sub_part}_") for answer_label in student_answers)
                    if not has_answer:
                        correct_pattern = f"part2_{question_num}_{sub_part}_{answer}"
                        matching_circles = find_matching_circles(all_circles, [correct_pattern])
                        missing_answers["part2"].extend(matching_circles)

        elif section_type == "ESSAY_CODE":
            # Part 3: Kiểm tra từng vị trí
            for question in questions:
                question_num = str(question.get("questionNumber", ""))
                answer_str = str(question.get("answer", ""))

                for position in range(1, len(answer_str) + 1):
                    # Kiểm tra vị trí này có ký tự nào được chọn không
                    has_answer = any(answer.startswith(f"part3_{question_num}_") and f"_{position}_" in answer for answer in student_answers)
                    if not has_answer:
                        # Tìm ký tự đúng ở vị trí này
                        char = answer_str[position - 1]
                        if char == '-':
                            symbol = 'minus'
                        elif char == ',':
                            symbol = 'comma'
                        elif char.isdigit():
                            symbol = char
                        else:
                            continue

                        correct_pattern = f"part3_{question_num}_{symbol}_{position}"
                        matching_circles = find_matching_circles(all_circles, [correct_pattern])
                        missing_answers["part3"].extend(matching_circles)

    return missing_answers


def check_missing_answers(student_answers: List[str], correct_answers, all_circles: List[str]) -> Dict[str, List[str]]:
    """
    Kiểm tra các câu không có đáp án nào được chọn
    Hỗ trợ cả format cũ (Dict) và format mới (List)

    Args:
        student_answers: List các circle labels mà học sinh đã tô
        correct_answers: Dict hoặc List chứa đáp án đúng
        all_circles: List tất cả circle labels có thể

    Returns:
        Dict với key là part và value là list các đáp án đúng bị thiếu
    """
    # Kiểm tra format mới (List) hay cũ (Dict)
    if isinstance(correct_answers, list):
        return check_missing_answers_new_format(student_answers, correct_answers, all_circles)

    # Format cũ (Dict)
    missing_answers = {
        "part1": [],
        "part2": [],
        "part3": []
    }

    # Part 1: Kiểm tra từng câu
    if "part1" in correct_answers:
        for question_num in correct_answers["part1"].keys():
            # Kiểm tra xem câu này có đáp án nào được chọn không
            has_answer = any(answer.startswith(f"part1_{question_num}_") for answer in student_answers)
            if not has_answer:
                # Tìm đáp án đúng cho câu này
                correct_pattern = f"part1_{question_num}_{correct_answers['part1'][question_num].lower()}"
                matching_circles = find_matching_circles(all_circles, [correct_pattern])
                missing_answers["part1"].extend(matching_circles)

    # Part 2: Kiểm tra từng sub-question
    if "part2" in correct_answers:
        for question_num, answers_data in correct_answers["part2"].items():
            if isinstance(answers_data, str):
                answers_list = answers_data.split(",")
                sub_parts = ["a", "b", "c", "d"]
                for i, answer in enumerate(answers_list):
                    if i < len(sub_parts):
                        sub_part = sub_parts[i]
                        # Kiểm tra sub-question này có đáp án không
                        has_answer = any(answer_label.startswith(f"part2_{question_num}_{sub_part}_") for answer_label in student_answers)
                        if not has_answer:
                            correct_pattern = f"part2_{question_num}_{sub_part}_{answer.strip()}"
                            matching_circles = find_matching_circles(all_circles, [correct_pattern])
                            missing_answers["part2"].extend(matching_circles)

            elif isinstance(answers_data, dict):
                for sub_part, answer in answers_data.items():
                    has_answer = any(answer_label.startswith(f"part2_{question_num}_{sub_part}_") for answer_label in student_answers)
                    if not has_answer:
                        correct_pattern = f"part2_{question_num}_{sub_part}_{answer.strip()}"
                        matching_circles = find_matching_circles(all_circles, [correct_pattern])
                        missing_answers["part2"].extend(matching_circles)

    # Part 3: Kiểm tra từng vị trí
    if "part3" in correct_answers:
        for question_num, answer_str in correct_answers["part3"].items():
            for position in range(1, len(answer_str) + 1):
                # Kiểm tra vị trí này có ký tự nào được chọn không
                has_answer = any(answer.startswith(f"part3_{question_num}_") and f"_{position}_" in answer for answer in student_answers)
                if not has_answer:
                    # Tìm ký tự đúng ở vị trí này
                    char = answer_str[position - 1]
                    if char == '-':
                        symbol = 'minus'
                    elif char == ',':
                        symbol = 'comma'
                    elif char.isdigit():
                        symbol = char
                    else:
                        continue

                    correct_pattern = f"part3_{question_num}_{symbol}_{position}"
                    matching_circles = find_matching_circles(all_circles, [correct_pattern])
                    missing_answers["part3"].extend(matching_circles)

    return missing_answers


def mark_correct_answers_on_image(image_path: str, exam_list: List[Dict[str, Any]], section_config: List[Dict[str, Any]] = None, output_dir: str = "output") -> Tuple[str, Dict[str, Any]]:
    """
    Đánh dấu đáp án đúng lên ảnh và trả về đường dẫn ảnh + đáp án học sinh + student ID

    Args:
        image_path: Đường dẫn ảnh đầu vào
        exam_list: List chứa các mã đề với đáp án đúng
        output_dir: Thư mục đầu ra

    Returns:
        Tuple (image_path, response_data)

    Raises:
        ValueError: Nếu không tìm thấy mã đề phù hợp
    """
    # Đọc ảnh
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Không thể đọc ảnh: {image_path}")

    # Phát hiện tất cả circles không cần debug images
    detection_results, _ = detect_circles(image_path, debug=False)
    all_circles = detection_results.get("all_answers", [])
    student_answers = detection_results.get("student_answers", [])  # Đáp án học sinh đã tô

    # Parse student ID và exam code từ student answers
    student_id = parse_student_id_from_answers(student_answers)
    exam_code = parse_exam_code_from_answers(student_answers)

    # Tìm mã đề phù hợp trong danh sách
    try:
        matched_exam = find_exam_code_in_list(exam_code, exam_list)
        correct_answers = matched_exam.get("answer_json", [])
        grading_session_id = matched_exam.get("grading_session_id", None)
    except ValueError as e:
        # Kiểm tra xem có phải định dạng cũ không (có mã đề "000")
        legacy_exam = None
        for exam in exam_list:
            if exam.get("code", "") == "000":
                legacy_exam = exam
                break

        if legacy_exam:
            # Sử dụng định dạng cũ
            print(f"🔄 Using legacy format as fallback for exam_code '{exam_code}'")
            matched_exam = legacy_exam
            correct_answers = matched_exam.get("answer_json", [])
            grading_session_id = matched_exam.get("grading_session_id", None)
        else:
            # Trả về lỗi nếu không tìm thấy mã đề
            return "", {
                "error": str(e),
                "student_code": student_id,
                "exam_code": exam_code,
                "available_exam_codes": [exam.get("code", "") for exam in exam_list]
            }

    # Sử dụng formatted_answers từ detect_circles thay vì parse lại
    student_answers_formatted = detection_results.get("formatted_answers", {
        "part1": {},
        "part2": {},
        "part3": {}
    })

    # Parse đáp án đúng
    marked_circles_patterns = parse_correct_answers(correct_answers)

    # Kiểm tra các trường hợp đặc biệt
    multiple_part1 = check_multiple_answers_part1(student_answers)
    multiple_part2 = check_multiple_answers_part2(student_answers)
    multiple_part3 = check_multiple_answers_part3(student_answers, correct_answers)
    missing_answers = check_missing_answers(student_answers, correct_answers, all_circles)

    # Đánh dấu các circles lên ảnh
    marked_image = image.copy()

    # Set để theo dõi các circles đã được đánh dấu
    marked_circles = set()

    # 1. Đánh dấu màu vàng cho các trường hợp nhiều đáp án
    # Part 1: Nhiều hơn 1 đáp án trong cùng câu
    for question_num, duplicate_answers in multiple_part1.items():
        for circle_label in duplicate_answers:
            x, y = extract_coordinates_from_label(circle_label)
            if x > 0 and y > 0:
                cv2.circle(marked_image, (x, y), 10, (0, 255, 255), 2)  # Yellow
                marked_circles.add(circle_label)

    # Part 2: Cả D và S trong cùng sub-question
    for key, duplicate_answers in multiple_part2.items():
        for circle_label in duplicate_answers:
            x, y = extract_coordinates_from_label(circle_label)
            if x > 0 and y > 0:
                cv2.circle(marked_image, (x, y), 10, (0, 255, 255), 2)  # Yellow
                marked_circles.add(circle_label)

    # Part 3: Ô thừa và nhiều ký tự cùng vị trí
    for circle_label in multiple_part3["extra_answers"]:
        x, y = extract_coordinates_from_label(circle_label)
        if x > 0 and y > 0:
            cv2.circle(marked_image, (x, y), 10, (0, 255, 255), 2)  # Yellow
            marked_circles.add(circle_label)

    for key, duplicate_answers in multiple_part3["multiple_at_position"].items():
        for circle_label in duplicate_answers:
            x, y = extract_coordinates_from_label(circle_label)
            if x > 0 and y > 0:
                cv2.circle(marked_image, (x, y), 10, (0, 255, 255), 2)  # Yellow
                marked_circles.add(circle_label)

    # 2. Đánh dấu màu đỏ cho đáp án đúng bị thiếu (không có ô nào được chọn)
    for part_name, missing_circles in missing_answers.items():
        for circle_label in missing_circles:
            if circle_label not in marked_circles:
                x, y = extract_coordinates_from_label(circle_label)
                if x > 0 and y > 0:
                    cv2.circle(marked_image, (x, y), 10, (0, 0, 255), 2)  # Red
                    marked_circles.add(circle_label)

    # 3. Đánh dấu màu xanh/đỏ cho các đáp án bình thường (không có vấn đề đặc biệt)
    for part_name, patterns in marked_circles_patterns.items():
        matching_circles = find_matching_circles(all_circles, patterns)

        for circle_label in matching_circles:
            if circle_label not in marked_circles:
                x, y = extract_coordinates_from_label(circle_label)
                if x > 0 and y > 0:
                    # Kiểm tra xem học sinh có tô đáp án này không
                    if circle_label in student_answers:
                        # Học sinh đã tô đúng -> màu xanh lá cây
                        color = (0, 255, 0)  # Green
                    else:
                        # Học sinh chưa tô hoặc tô sai -> màu đỏ
                        color = (0, 0, 255)  # Red

                    # Vẽ vòng tròn với màu tương ứng
                    cv2.circle(marked_image, (x, y), 10, color, 2)
                    marked_circles.add(circle_label)

    # Tạo summary trước để có correct_matches
    summary_data = create_answer_summary(correct_answers, all_circles, student_answers)

    # Tính điểm dựa trên section config
    if section_config:
        scores = calculate_scores(
            summary_data.get("correct_matches", {"part1": [], "part2": [], "part3": []}),
            summary_data.get("incorrect_missing", {"part1": [], "part2": [], "part3": []}),
            multiple_part1,
            multiple_part2,
            multiple_part3,
            student_answers_formatted,
            correct_answers,
            section_config
        )

        # Ghi thông tin điểm lên ảnh
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        color = (0, 0, 255)  # Màu đỏ
        thickness = 2

        # Vị trí ghi text (góc trên bên trái)
        y_offset = 30
        x_offset = 20

        # Tổng điểm
        total_text = f"Tong diem: {scores['total_score']:.2f}/{scores['max_score']}"
        cv2.putText(marked_image, total_text, (x_offset, y_offset), font, font_scale, color, thickness)

        # Số câu đúng từng phần
        y_offset += 30
        part1_text = f"Phan 1: {scores['part1']['correct_count']}/{scores['part1']['total_questions']} cau ({scores['part1']['score']:.2f}d)"
        cv2.putText(marked_image, part1_text, (x_offset, y_offset), font, font_scale, color, thickness)

        # Phần 2 chi tiết hơn
        y_offset += 30
        part2_basic = f"Phan 2: {scores['part2']['correct_count']}/{scores['part2']['total_questions']} cau hoan toan dung ({scores['part2']['score']:.2f}d)"
        cv2.putText(marked_image, part2_basic, (x_offset, y_offset), font, font_scale, color, thickness)

        # Hiển thị chi tiết từng câu phần 2 nếu có thông tin
        if 'part2_details' in scores:
            for i, detail in enumerate(scores['part2_details']):  # Hiển thị tất cả các câu
                y_offset += 25
                detail_text = f"  C{detail['question']}: {detail['correct_count']}/4 dung -> {detail['score']:.2f}d"
                cv2.putText(marked_image, detail_text, (x_offset + 10, y_offset), font, 0.6, color, thickness)

        y_offset += 30
        part3_text = f"Phan 3: {scores['part3']['correct_count']}/{scores['part3']['total_questions']} cau ({scores['part3']['score']:.2f}d)"
        cv2.putText(marked_image, part3_text, (x_offset, y_offset), font, font_scale, color, thickness)

        print(f"📊 Calculated scores: {scores}")
    else:
        scores = None
        print("⚠️ No section config provided, skipping score calculation")

    # Tạo thư mục output nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)

    # Lưu ảnh đánh dấu vào thư mục output
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    timestamp = int(time.time())
    marked_image_filename = f"{base_name}_marked_{timestamp}.png"
    marked_image_path = os.path.join(output_dir, marked_image_filename)

    cv2.imwrite(marked_image_path, marked_image)
    print(f"💾 Saved marked image to: {marked_image_path}")

    # Chuyển đổi sang format mới với đường dẫn ảnh thay vì base64
    new_format_data = convert_to_new_format(
        student_answers_formatted,
        student_id,
        exam_code,
        marked_image_path  # Trả về đường dẫn thay vì base64
    )

    # Tạo báo cáo chi tiết về các vấn đề
    marking_report = create_marking_report(multiple_part1, multiple_part2, multiple_part3, missing_answers)

    # Thêm thông tin về các vấn đề được phát hiện
    new_format_data["marking_issues"] = {
        "report": marking_report,
        "raw_data": {
            "multiple_answers_part1": multiple_part1,
            "multiple_answers_part2": multiple_part2,
            "multiple_answers_part3": multiple_part3,
            "missing_answers": missing_answers
        },
        "total_marked_circles": len(marked_circles)
    }

    # Thêm thông tin về grading session
    new_format_data["grading_session_id"] = matched_exam.get("grading_session_id", None)
    new_format_data["matched_exam_code"] = matched_exam.get("code", "")

    # Thêm thông tin điểm số nếu có
    if section_config and 'scores' in locals():
        new_format_data["scores"] = scores

    # Chỉ giữ lại ảnh đánh dấu chính (không có debug images)
    # Ảnh đánh dấu đã được lưu vào thư mục output

    return marked_image_path, new_format_data


def create_marking_report(multiple_part1: Dict[str, List[str]],
                         multiple_part2: Dict[str, List[str]],
                         multiple_part3: Dict[str, List[str]],
                         missing_answers: Dict[str, List[str]]) -> Dict[str, Any]:
    """
    Tạo báo cáo chi tiết về các vấn đề đánh dấu

    Args:
        multiple_part1: Dict các câu Part 1 có nhiều đáp án
        multiple_part2: Dict các câu Part 2 có nhiều đáp án
        multiple_part3: Dict các vấn đề Part 3
        missing_answers: Dict các đáp án bị thiếu

    Returns:
        Dict chứa báo cáo chi tiết
    """
    report = {
        "summary": {
            "total_issues": 0,
            "part1_multiple_answers": len(multiple_part1),
            "part2_multiple_answers": len(multiple_part2),
            "part3_extra_answers": len(multiple_part3.get("extra_answers", [])),
            "part3_multiple_at_position": len(multiple_part3.get("multiple_at_position", {})),
            "missing_answers_total": sum(len(circles) for circles in missing_answers.values())
        },
        "details": {
            "part1_issues": [],
            "part2_issues": [],
            "part3_issues": [],
            "missing_answers_details": missing_answers
        }
    }

    # Chi tiết Part 1
    for question_num, duplicate_answers in multiple_part1.items():
        report["details"]["part1_issues"].append({
            "question": question_num,
            "issue": "multiple_answers",
            "description": f"Câu {question_num} có {len(duplicate_answers)} đáp án được chọn",
            "affected_circles": duplicate_answers
        })

    # Chi tiết Part 2
    for key, duplicate_answers in multiple_part2.items():
        question_num, sub_part = key.split("_")
        report["details"]["part2_issues"].append({
            "question": f"{question_num}{sub_part}",
            "issue": "both_d_and_s",
            "description": f"Câu {question_num}{sub_part} có cả D và S được chọn",
            "affected_circles": duplicate_answers
        })

    # Chi tiết Part 3
    for circle_label in multiple_part3.get("extra_answers", []):
        parts = circle_label.split("_")
        if len(parts) >= 4:
            question_num = parts[1]
            position = parts[3]
            report["details"]["part3_issues"].append({
                "question": question_num,
                "issue": "extra_answer",
                "description": f"Câu {question_num} có ô thừa ở vị trí {position}",
                "affected_circles": [circle_label]
            })

    for key, duplicate_answers in multiple_part3.get("multiple_at_position", {}).items():
        question_num, position = key.split("_")
        report["details"]["part3_issues"].append({
            "question": question_num,
            "issue": "multiple_at_position",
            "description": f"Câu {question_num} có nhiều ký tự ở vị trí {position}",
            "affected_circles": duplicate_answers
        })

    # Tính tổng số vấn đề
    report["summary"]["total_issues"] = (
        len(multiple_part1) +
        len(multiple_part2) +
        len(multiple_part3.get("extra_answers", [])) +
        len(multiple_part3.get("multiple_at_position", {})) +
        report["summary"]["missing_answers_total"]
    )

    return report


def calculate_scores(correct_matches: Dict[str, List[str]],
                    incorrect_missing: Dict[str, List[str]],
                    multiple_answers_part1: Dict[str, List[str]],
                    multiple_answers_part2: Dict[str, List[str]],
                    multiple_answers_part3: Dict[str, List[str]],
                    student_answers_formatted: Dict[str, Dict[str, str]],
                    correct_answers: List[Dict[str, Any]],
                    section_config: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Tính điểm dựa trên section config và kết quả chấm

    Args:
        correct_matches: Các đáp án đúng (màu xanh)
        incorrect_missing: Các đáp án sai/thiếu (màu đỏ)
        multiple_answers_part1: Câu có nhiều đáp án part 1 (màu vàng)
        multiple_answers_part2: Câu có nhiều đáp án part 2 (màu vàng)
        multiple_answers_part3: Câu có nhiều đáp án part 3 (màu vàng)
        student_answers_formatted: Đáp án học sinh đã format
        correct_answers: Đáp án đúng
        section_config: Cấu hình chấm điểm

    Returns:
        Dict chứa thông tin điểm số
    """
    if not section_config:
        section_config = []

    scores = {
        "part1": {"correct_count": 0, "total_questions": 0, "score": 0.0, "points_per_question": 0.0},
        "part2": {"correct_count": 0, "total_questions": 0, "score": 0.0, "rule": {}},
        "part3": {"correct_count": 0, "total_questions": 0, "score": 0.0, "points_per_question": 0.0},
        "total_score": 0.0,
        "max_score": 10.0
    }

    # Lấy config cho từng section
    section_configs = {}
    for config in section_config:
        section_type = config.get("sectionType", "")
        section_order = config.get("sectionOrder", 0)
        if section_type == "MULTIPLE_CHOICE":
            section_configs["part1"] = config
        elif section_type == "TRUE_FALSE":
            section_configs["part2"] = config
        elif section_type == "ESSAY":
            section_configs["part3"] = config

    # Tính điểm Part 1 (MULTIPLE_CHOICE)
    if "part1" in section_configs:
        config = section_configs["part1"]
        points_per_question = config.get("pointsPerQuestion", 0.25)
        question_count = config.get("questionCount", 0)

        scores["part1"]["points_per_question"] = points_per_question
        scores["part1"]["total_questions"] = question_count

        # Đếm câu đúng (không có trong multiple_answers và có trong correct_matches)
        correct_count = 0
        for section in correct_answers:
            if section.get("sectionType") == "MULTIPLE_CHOICE":
                questions = section.get("questions", [])
                for question in questions:
                    question_num = str(question.get("questionNumber", ""))

                    # Kiểm tra câu này có bị multiple answers không
                    if question_num not in multiple_answers_part1:
                        # Kiểm tra có đáp án đúng không
                        student_answer = student_answers_formatted.get("part1", {}).get(question_num, "")
                        correct_answer = question.get("answer", "")

                        if student_answer.upper() == correct_answer.upper():
                            correct_count += 1

        scores["part1"]["correct_count"] = correct_count
        scores["part1"]["score"] = correct_count * points_per_question

    # Tính điểm Part 2 (TRUE_FALSE)
    if "part2" in section_configs:
        config = section_configs["part2"]
        rule = config.get("rule", {})
        question_count = config.get("questionCount", 0)

        scores["part2"]["rule"] = rule
        scores["part2"]["total_questions"] = question_count

        # Debug: Xem cấu trúc student_answers_formatted (comment out for production)
        # print(f"🔍 student_answers_formatted part2: {student_answers_formatted.get('part2', {})}")

        # Tính điểm cho từng câu và lưu chi tiết
        total_score_part2 = 0.0
        correct_questions = 0
        part2_details = []

        for section in correct_answers:
            if section.get("sectionType") == "TRUE_FALSE":
                questions = section.get("questions", [])
                for question in questions:
                    question_num = str(question.get("questionNumber", ""))
                    correct_answer_dict = question.get("answer", {})

                    # Kiểm tra câu này có bị multiple answers không
                    question_has_multiple = any(key.startswith(f"{question_num}_") for key in multiple_answers_part2.keys())

                    if not question_has_multiple:
                        # Đếm số câu con đúng từ correct_matches
                        correct_sub_count = 0

                        for sub_part in ["a", "b", "c", "d"]:
                            # Tìm trong correct_matches xem có circle nào match với pattern này không
                            # Pattern: part2_{question_num}_{sub_part}_{answer}_{x}_{y}
                            correct_answer = correct_answer_dict.get(sub_part, "")
                            normalized_correct = "D" if correct_answer == "Đ" else "S" if correct_answer == "S" else correct_answer

                            # Tìm trong correct_matches
                            pattern = f"part2_{question_num}_{sub_part}_{normalized_correct}_"
                            found_match = any(match.startswith(pattern) for match in correct_matches.get("part2", []))

                            # Debug logging (comment out for production)
                            # print(f"   🔍 Q{question_num}{sub_part}: Looking for pattern '{pattern}' in {correct_matches.get('part2', [])} -> {found_match}")

                            if found_match:
                                correct_sub_count += 1

                        # Áp dụng rule để tính điểm cho câu này
                        question_score = rule.get(str(correct_sub_count), 0.0)
                        total_score_part2 += question_score

                        # Debug logging (comment out for production)
                        # print(f"🔍 Part2 Question {question_num}: {correct_sub_count}/4 correct, score: {question_score}")
                        # print(f"   Debug: {debug_info}")

                        # Lưu chi tiết câu này
                        part2_details.append({
                            "question": question_num,
                            "correct_count": correct_sub_count,
                            "total_sub_questions": 4,
                            "score": question_score
                        })

                        if correct_sub_count == 4:  # Câu hoàn toàn đúng
                            correct_questions += 1
                    else:
                        # Câu có multiple answers = 0 điểm
                        part2_details.append({
                            "question": question_num,
                            "correct_count": 0,
                            "total_sub_questions": 4,
                            "score": 0.0,
                            "note": "Multiple answers"
                        })

        scores["part2"]["correct_count"] = correct_questions
        scores["part2"]["score"] = total_score_part2
        scores["part2_details"] = part2_details

    # Tính điểm Part 3 (ESSAY)
    if "part3" in section_configs:
        config = section_configs["part3"]
        points_per_question = config.get("pointsPerQuestion", 0.5)
        question_count = config.get("questionCount", 0)

        scores["part3"]["points_per_question"] = points_per_question
        scores["part3"]["total_questions"] = question_count

        # Đếm câu đúng (không có màu vàng/đỏ, chỉ có màu xanh)
        correct_count = 0
        for section in correct_answers:
            if section.get("sectionType") == "ESSAY_CODE":
                questions = section.get("questions", [])
                for question in questions:
                    question_num = str(question.get("questionNumber", ""))

                    # Kiểm tra câu này có vấn đề không (multiple answers hoặc extra answers)
                    has_issues = (
                        question_num in multiple_answers_part3.get("multiple_at_position", {}) or
                        any(answer.startswith(f"part3_{question_num}_") for answer in multiple_answers_part3.get("extra_answers", []))
                    )

                    if not has_issues:
                        # Kiểm tra tất cả ký tự có đúng không
                        correct_answer = str(question.get("answer", ""))
                        student_answer = student_answers_formatted.get("part3", {}).get(question_num, "")

                        if student_answer == correct_answer:
                            correct_count += 1

        scores["part3"]["correct_count"] = correct_count
        scores["part3"]["score"] = correct_count * points_per_question

    # Tính tổng điểm
    scores["total_score"] = scores["part1"]["score"] + scores["part2"]["score"] + scores["part3"]["score"]

    return scores


def create_answer_summary(correct_answers, all_circles: List[str], student_answers: List[str] = None) -> Dict[str, Any]:
    """
    Tạo summary thông tin về việc đánh dấu đáp án
    Hỗ trợ cả format cũ (Dict) và format mới (List)

    Args:
        correct_answers: Dict hoặc List chứa đáp án đúng
        all_circles: List tất cả circle labels
        student_answers: List đáp án học sinh đã tô

    Returns:
        Dict chứa thông tin summary
    """
    if student_answers is None:
        student_answers = []

    marked_circles_patterns = parse_correct_answers(correct_answers)
    
    summary = {
        "total_questions": {},
        "marked_circles": {},
        "unmarked_patterns": {},
        "correct_matches": {},  # Đáp án đúng và học sinh đã tô
        "incorrect_missing": {}  # Đáp án đúng nhưng học sinh chưa tô
    }
    
    for part, patterns in marked_circles_patterns.items():
        matching_circles = find_matching_circles(all_circles, patterns)
        
        summary["total_questions"][part] = len(patterns)
        summary["marked_circles"][part] = len(matching_circles)
        
        # Phân loại đáp án
        correct_matches = []
        incorrect_missing = []
        
        for circle in matching_circles:
            if circle in student_answers:
                correct_matches.append(circle)
            else:
                incorrect_missing.append(circle)
        
        summary["correct_matches"][part] = correct_matches
        summary["incorrect_missing"][part] = incorrect_missing
        
        # Tìm patterns không khớp
        unmarked = []
        for pattern in patterns:
            found = False
            for circle in all_circles:
                if circle.startswith(pattern + "_"):
                    found = True
                    break
            if not found:
                unmarked.append(pattern)
        
        summary["unmarked_patterns"][part] = unmarked
    
    return summary
