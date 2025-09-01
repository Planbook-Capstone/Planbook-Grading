import cv2
import numpy as np
import os
import json
from typing import Dict, List, Any, Tuple
from api.circle_detection_service import detect_circles


def parse_correct_answers(correct_answers: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Parse correct answers từ JSON format thành danh sách các circle labels cần đánh dấu
    
    Args:
        correct_answers: Dict chứa đáp án đúng cho từng phần
        
    Returns:
        Dict với key là part và value là list các circle labels cần đánh dấu
    """
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
        for question_num, answers_str in correct_answers["part2"].items():
            # answers_str format: "D,D,S,S" tương ứng với a,b,c,d
            answers_list = answers_str.split(",")
            sub_parts = ["a", "b", "c", "d"]
            
            for i, answer in enumerate(answers_list):
                if i < len(sub_parts):
                    sub_part = sub_parts[i]
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


def mark_correct_answers_on_image(image_path: str, correct_answers: Dict[str, Any], output_dir: str = "result") -> str:
    """
    Đánh dấu đáp án đúng lên ảnh
    
    Args:
        image_path: Đường dẫn ảnh đầu vào
        correct_answers: Dict chứa đáp án đúng
        output_dir: Thư mục đầu ra
        
    Returns:
        Đường dẫn ảnh đã đánh dấu
    """
    # Tạo thư mục output nếu chưa có
    os.makedirs(output_dir, exist_ok=True)
    
    # Đọc ảnh
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Không thể đọc ảnh: {image_path}")
    
    # Phát hiện tất cả circles
    detection_results, _ = detect_circles(image_path, debug=False)
    all_circles = detection_results.get("all_answers", [])
    student_answers = detection_results.get("student_answers", [])  # Đáp án học sinh đã tô
    
    # Parse đáp án đúng
    marked_circles_patterns = parse_correct_answers(correct_answers)
    
    # Tìm các circles cần đánh dấu cho từng part
    circles_to_mark = []
    
    for part, patterns in marked_circles_patterns.items():
        matching_circles = find_matching_circles(all_circles, patterns)
        circles_to_mark.extend(matching_circles)
    
    # Đánh dấu các circles lên ảnh
    marked_image = image.copy()
    
    for circle_label in circles_to_mark:
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
  
           
    
    # Lưu ảnh đã đánh dấu
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_marked_answers.png")
    cv2.imwrite(output_path, marked_image)
    
    return output_path


def create_answer_summary(correct_answers: Dict[str, Any], all_circles: List[str], student_answers: List[str] = None) -> Dict[str, Any]:
    """
    Tạo summary thông tin về việc đánh dấu đáp án
    
    Args:
        correct_answers: Dict chứa đáp án đúng
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
