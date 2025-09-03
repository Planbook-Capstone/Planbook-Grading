import cv2
import numpy as np
import math
import base64
import os
import time
from typing import List, Dict, Tuple, Any, Optional
import imutils


def find_optimal_threshold_for_squares(gray_image: np.ndarray) -> Tuple[int, np.ndarray]:
    """
    Tìm threshold tối ưu để detect ô vuông đen bằng cách thử nhiều giá trị threshold

    Args:
        gray_image: Ảnh grayscale

    Returns:
        Tuple (best_threshold, best_thresh_image)
    """
    print("🔍 Tìm threshold tối ưu cho detection ô vuông đen...")

    best_threshold = -1
    best_count = 0
    best_thresh_image = None
    results = []

    for threshold_val in range(20, 201, 10):
        _, thresh = cv2.threshold(gray_image, threshold_val, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        valid_squares = count_valid_black_squares(contours)
        results.append({'thresh': threshold_val, 'squares': valid_squares})

        if valid_squares > best_count:
            best_count = valid_squares
            best_threshold = threshold_val
            best_thresh_image = thresh.copy()
            print(f"🎯 Threshold mới tốt nhất: {threshold_val}, squares={valid_squares}")

    print("\n📊 Kết quả tìm threshold:")
    for res in results:
        status = "🎯 BEST" if res['thresh'] == best_threshold else "✅" if res['squares'] > 0 else "❌"
        print(f"   Threshold {res['thresh']}: {res['squares']} squares {status}")

    if best_threshold == -1:
        # Fallback: sử dụng threshold 128 nếu không tìm thấy gì
        print("⚠️ Không tìm thấy threshold tối ưu, sử dụng fallback threshold=128")
        best_threshold = 128
        _, best_thresh_image = cv2.threshold(gray_image, best_threshold, 255, cv2.THRESH_BINARY_INV)

    print(f"🎯 Threshold tối ưu: {best_threshold} với {best_count} ô vuông")
    return best_threshold, best_thresh_image


def count_valid_black_squares(contours) -> int:
    """
    Đếm số lượng ô vuông đen hợp lệ từ danh sách contours

    Args:
        contours: Danh sách contours từ cv2.findContours

    Returns:
        int: Số lượng ô vuông đen hợp lệ
    """
    valid_squares = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Điều chỉnh ngưỡng area để cân bằng giữa ô lớn và ô nhỏ
        if area < 20 or area > 3000:  # Từ 20 pixels (4x5) đến 3000 pixels
            continue

        # Sử dụng epsilon nhỏ để chính xác hơn
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # Chấp nhận cả contour không hoàn hảo để detect ô vuông nhỏ
        if len(approx) >= 3:  # Chấp nhận từ 3 cạnh trở lên
            _, _, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h

            # Tính các metrics
            bbox_area = w * h
            contour_area = cv2.contourArea(cnt)
            fill_ratio = contour_area / bbox_area if bbox_area > 0 else 0

            perimeter = cv2.arcLength(cnt, True)
            compactness = 4 * np.pi * contour_area / (perimeter * perimeter) if perimeter > 0 else 0

            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = contour_area / hull_area if hull_area > 0 else 0

            # Điều kiện để detect ô vuông đen, tránh hình tròn
            if (0.85 <= aspect_ratio <= 1.15 and  # Aspect ratio chặt cho hình vuông
                fill_ratio >= 0.8 and             # Fill ratio cao
                0.6 <= compactness <= 0.85 and    # Compactness của hình vuông (tránh hình tròn ~1.0)
                solidity >= 0.85):                # Solidity cao
                valid_squares += 1

    return valid_squares


# Các function cũ đã được thay thế bằng detect_all_black_squares_direct


# Các function cũ đã được thay thế bằng detect_all_black_squares_direct


def calculate_student_id_roi(detected_squares: List[Dict[str, Any]]) -> Tuple[int, int, int, int]:
    """
    Tính toán ROI cho student ID dựa trên các ô vuông 3, 9, 10
    - Chiều dài: từ góc phải dưới ô 3 -> góc phải trên ô 10
    - Chiều rộng: từ góc phải dưới ô 3 -> góc trái trên ô 9

    Args:
        detected_squares: Danh sách các ô vuông đã detect

    Returns:
        Tuple (x, y, width, height) của ROI student ID
    """
    # Tìm các ô vuông cần thiết
    square_3 = None
    square_9 = None
    square_10 = None

    for square in detected_squares:
        if square["id"] == 3:
            square_3 = square
        elif square["id"] == 9:
            square_9 = square
        elif square["id"] == 10:
            square_10 = square

    if not all([square_3, square_9, square_10]):
        missing = []
        if not square_3: missing.append("3")
        if not square_9: missing.append("9")
        if not square_10: missing.append("10")
        print(f"⚠️ Thiếu ô vuông ID: {missing} cho student ID ROI")
        return (0, 0, 0, 0)

    # Lấy bounding box của các ô vuông
    bbox_3 = square_3["bounding_box"]  # (x, y, w, h)
    bbox_9 = square_9["bounding_box"]
    bbox_10 = square_10["bounding_box"]

    # Tính toán các góc
    # Góc phải dưới ô 3
    bottom_right_3 = (bbox_3[0] + bbox_3[2], bbox_3[1] + bbox_3[3])

    # Góc phải trên ô 10
    top_right_10 = (bbox_10[0] + bbox_10[2], bbox_10[1])

    # Góc trái trên ô 9
    top_left_9 = (bbox_9[0], bbox_9[1])

    # Tính toán ROI
    # Góc trái trên ROI: lấy min của x và y
    roi_x = min(top_left_9[0], bottom_right_3[0], top_right_10[0])
    roi_y = min(top_left_9[1], bottom_right_3[1], top_right_10[1])

    # Góc phải dưới ROI: lấy max của x và y
    roi_right = max(top_left_9[0], bottom_right_3[0], top_right_10[0])
    roi_bottom = max(top_left_9[1], bottom_right_3[1], top_right_10[1])

    # Tính chiều rộng và chiều cao
    roi_width = roi_right - roi_x
    roi_height = roi_bottom - roi_y

    print(f"📍 Student ID ROI: ({roi_x}, {roi_y}, {roi_width}, {roi_height})")
    print(f"   Góc phải dưới ô 3: {bottom_right_3}")
    print(f"   Góc phải trên ô 10: {top_right_10}")
    print(f"   Góc trái trên ô 9: {top_left_9}")

    return (roi_x, roi_y, roi_width, roi_height)


def calculate_exam_code_roi(detected_squares: List[Dict[str, Any]]) -> Tuple[int, int, int, int]:
    """
    Tính toán ROI cho exam code dựa trên các ô vuông 5, 9, 18
    - Chiều dài: từ góc trái dưới ô 5 -> góc trái trên ô 18
    - Chiều rộng: từ góc trái dưới ô 5 -> góc phải trên ô 9

    Args:
        detected_squares: Danh sách các ô vuông đã detect

    Returns:
        Tuple (x, y, width, height) của ROI exam code
    """
    # Tìm các ô vuông cần thiết
    square_5 = None
    square_9 = None
    square_18 = None

    for square in detected_squares:
        if square["id"] == 5:
            square_5 = square
        elif square["id"] == 9:
            square_9 = square
        elif square["id"] == 18:
            square_18 = square

    if not all([square_5, square_9, square_18]):
        missing = []
        if not square_5: missing.append("5")
        if not square_9: missing.append("9")
        if not square_18: missing.append("18")
        print(f"⚠️ Thiếu ô vuông ID: {missing} cho exam code ROI")
        return (0, 0, 0, 0)

    # Lấy bounding box của các ô vuông
    bbox_5 = square_5["bounding_box"]  # (x, y, w, h)
    bbox_9 = square_9["bounding_box"]
    bbox_18 = square_18["bounding_box"]

    # Tính toán các góc
    # Góc trái dưới ô 5
    bottom_left_5 = (bbox_5[0], bbox_5[1] + bbox_5[3])

    # Góc trái trên ô 18
    top_left_18 = (bbox_18[0], bbox_18[1])

    # Góc phải trên ô 9
    top_right_9 = (bbox_9[0] + bbox_9[2], bbox_9[1])

    # Tính toán ROI
    # Góc trái trên ROI: lấy min của x và y
    roi_x = min(bottom_left_5[0], top_left_18[0], top_right_9[0])
    roi_y = min(bottom_left_5[1], top_left_18[1], top_right_9[1])

    # Góc phải dưới ROI: lấy max của x và y
    roi_right = max(bottom_left_5[0], top_left_18[0], top_right_9[0])
    roi_bottom = max(bottom_left_5[1], top_left_18[1], top_right_9[1])

    # Tính chiều rộng và chiều cao
    roi_width = roi_right - roi_x
    roi_height = roi_bottom - roi_y

    print(f"📍 Exam Code ROI: ({roi_x}, {roi_y}, {roi_width}, {roi_height})")
    print(f"   Ô 5 bbox: {bbox_5} -> Góc trái dưới: {bottom_left_5}")
    print(f"   Ô 18 bbox: {bbox_18} -> Góc trái trên: {top_left_18}")
    print(f"   Ô 9 bbox: {bbox_9} -> Góc phải trên: {top_right_9}")

    return (roi_x, roi_y, roi_width, roi_height)


def calculate_part1_roi(detected_squares: List[Dict[str, Any]]) -> Tuple[int, int, int, int]:
    """
    Tính toán ROI cho Part 1 dựa trên các ô vuông 2, 4, 7, 18
    - Chiều rộng: từ góc phải dưới ô 2 -> trái dưới ô 18
    - Chiều dài: từ góc phải dưới ô 4 -> trái trên ô 7

    Args:
        detected_squares: Danh sách các ô vuông đã detect

    Returns:
        Tuple (x, y, width, height) của ROI Part 1
    """
    # Tìm các ô vuông cần thiết
    square_2 = None
    square_4 = None
    square_7 = None
    square_18 = None

    for square in detected_squares:
        if square["id"] == 2:
            square_2 = square
        elif square["id"] == 4:
            square_4 = square
        elif square["id"] == 7:
            square_7 = square
        elif square["id"] == 18:
            square_18 = square

    if not all([square_2, square_4, square_7, square_18]):
        missing = []
        if not square_2: missing.append("2")
        if not square_4: missing.append("4")
        if not square_7: missing.append("7")
        if not square_18: missing.append("18")
        print(f"⚠️ Thiếu ô vuông ID: {missing} cho Part 1 ROI")
        return (0, 0, 0, 0)

    # Lấy bounding box của các ô vuông
    bbox_2 = square_2["bounding_box"]  # (x, y, w, h)
    bbox_4 = square_4["bounding_box"]
    bbox_7 = square_7["bounding_box"]
    bbox_18 = square_18["bounding_box"]

    # Tính toán các góc
    # Góc phải dưới ô 2
    bottom_right_2 = (bbox_2[0] + bbox_2[2], bbox_2[1] + bbox_2[3])

    # Góc phải dưới ô 4
    bottom_right_4 = (bbox_4[0] + bbox_4[2], bbox_4[1] + bbox_4[3])

    # Góc trái dưới ô 18
    bottom_left_18 = (bbox_18[0], bbox_18[1] + bbox_18[3])

    # Góc trái trên ô 7
    top_left_7 = (bbox_7[0], bbox_7[1])

    # Tính toán ROI dựa trên yêu cầu cụ thể
    # Chiều rộng: từ góc phải dưới ô 2 -> trái dưới ô 18
    roi_x = min(bottom_right_2[0], bottom_left_18[0])
    roi_width = abs(bottom_left_18[0] - bottom_right_2[0])

    # Chiều dài: từ góc phải dưới ô 4 -> trái trên ô 7
    roi_y = min(bottom_right_4[1], top_left_7[1])
    roi_height = abs(bottom_right_4[1] - top_left_7[1])

    print(f"📍 Part 1 ROI: ({roi_x}, {roi_y}, {roi_width}, {roi_height})")
    print(f"   Chiều rộng: Ô 2 phải dưới {bottom_right_2} -> Ô 18 trái dưới {bottom_left_18}")
    print(f"   Chiều dài: Ô 4 phải dưới {bottom_right_4} -> Ô 7 trái trên {top_left_7}")

    return (roi_x, roi_y, roi_width, roi_height)


def calculate_part2_roi(detected_squares: List[Dict[str, Any]]) -> Tuple[int, int, int, int]:
    """
    Tính toán ROI cho Part 2 dựa trên các ô vuông 2, 8, 13, 18
    - Chiều rộng: giữ nguyên như Part 1 (từ góc phải dưới ô 2 -> trái dưới ô 18)
    - Chiều dài: từ trái dưới ô 8 -> trái trên ô 13

    Args:
        detected_squares: Danh sách các ô vuông đã detect

    Returns:
        Tuple (x, y, width, height) của ROI Part 2
    """
    # Tìm các ô vuông cần thiết
    square_2 = None
    square_8 = None
    square_13 = None
    square_18 = None

    for square in detected_squares:
        if square["id"] == 2:
            square_2 = square
        elif square["id"] == 8:
            square_8 = square
        elif square["id"] == 13:
            square_13 = square
        elif square["id"] == 18:
            square_18 = square

    if not all([square_2, square_8, square_13, square_18]):
        missing = []
        if not square_2: missing.append("2")
        if not square_8: missing.append("8")
        if not square_13: missing.append("13")
        if not square_18: missing.append("18")
        print(f"⚠️ Thiếu ô vuông ID: {missing} cho Part 2 ROI")
        return (0, 0, 0, 0)

    # Lấy bounding box của các ô vuông
    bbox_2 = square_2["bounding_box"]  # (x, y, w, h)
    bbox_8 = square_8["bounding_box"]
    bbox_13 = square_13["bounding_box"]
    bbox_18 = square_18["bounding_box"]

    # Tính toán các góc
    # Góc phải dưới ô 2
    bottom_right_2 = (bbox_2[0] + bbox_2[2], bbox_2[1] + bbox_2[3])

    # Góc trái dưới ô 8
    bottom_left_8 = (bbox_8[0], bbox_8[1] + bbox_8[3])

    # Góc trái dưới ô 18
    bottom_left_18 = (bbox_18[0], bbox_18[1] + bbox_18[3])

    # Góc trái trên ô 13
    top_left_13 = (bbox_13[0], bbox_13[1])

    # Tính toán ROI dựa trên yêu cầu cụ thể
    # Chiều rộng: giữ nguyên như Part 1 (từ góc phải dưới ô 2 -> trái dưới ô 18)
    roi_x = min(bottom_right_2[0], bottom_left_18[0])
    roi_width = abs(bottom_left_18[0] - bottom_right_2[0])

    # Chiều dài: từ trái dưới ô 8 -> trái trên ô 13
    roi_y = min(bottom_left_8[1], top_left_13[1])
    roi_height = abs(bottom_left_8[1] - top_left_13[1])

    print(f"📍 Part 2 ROI: ({roi_x}, {roi_y}, {roi_width}, {roi_height})")
    print(f"   Chiều rộng: Ô 2 phải dưới {bottom_right_2} -> Ô 18 trái dưới {bottom_left_18}")
    print(f"   Chiều dài: Ô 8 trái dưới {bottom_left_8} -> Ô 13 trái trên {top_left_13}")

    return (roi_x, roi_y, roi_width, roi_height)


def calculate_part3_roi(detected_squares: List[Dict[str, Any]]) -> Tuple[int, int, int, int]:
    """
    Tính toán ROI cho Part 3 dựa trên các ô vuông 2, 11, 18, 23
    - Chiều rộng: giữ nguyên như Part 1 (từ góc phải dưới ô 2 -> trái dưới ô 18)
    - Chiều dài: từ trái dưới ô 11 -> trái trên ô 23

    Args:
        detected_squares: Danh sách các ô vuông đã detect

    Returns:
        Tuple (x, y, width, height) của ROI Part 3
    """
    # Tìm các ô vuông cần thiết
    square_2 = None
    square_11 = None
    square_18 = None
    square_23 = None

    for square in detected_squares:
        if square["id"] == 2:
            square_2 = square
        elif square["id"] == 11:
            square_11 = square
        elif square["id"] == 18:
            square_18 = square
        elif square["id"] == 23:
            square_23 = square

    if not all([square_2, square_11, square_18, square_23]):
        missing = []
        if not square_2: missing.append("2")
        if not square_11: missing.append("11")
        if not square_18: missing.append("18")
        if not square_23: missing.append("23")
        print(f"⚠️ Thiếu ô vuông ID: {missing} cho Part 3 ROI")
        return (0, 0, 0, 0)

    # Lấy bounding box của các ô vuông
    bbox_2 = square_2["bounding_box"]  # (x, y, w, h)
    bbox_11 = square_11["bounding_box"]
    bbox_18 = square_18["bounding_box"]
    bbox_23 = square_23["bounding_box"]

    # Tính toán các góc
    # Góc phải dưới ô 2
    bottom_right_2 = (bbox_2[0] + bbox_2[2], bbox_2[1] + bbox_2[3])

    # Góc trái dưới ô 11
    bottom_left_11 = (bbox_11[0], bbox_11[1] + bbox_11[3])

    # Góc trái dưới ô 18
    bottom_left_18 = (bbox_18[0], bbox_18[1] + bbox_18[3])

    # Góc trái trên ô 23
    top_left_23 = (bbox_23[0], bbox_23[1])

    # Tính toán ROI dựa trên yêu cầu cụ thể
    # Chiều rộng: giữ nguyên như Part 1 (từ góc phải dưới ô 2 -> trái dưới ô 18)
    roi_x = min(bottom_right_2[0], bottom_left_18[0])
    roi_width = abs(bottom_left_18[0] - bottom_right_2[0])

    # Chiều dài: từ trái dưới ô 11 -> trái trên ô 23
    roi_y = min(bottom_left_11[1], top_left_23[1])
    roi_height = abs(bottom_left_11[1] - top_left_23[1])

    print(f"📍 Part 3 ROI: ({roi_x}, {roi_y}, {roi_width}, {roi_height})")
    print(f"   Chiều rộng: Ô 2 phải dưới {bottom_right_2} -> Ô 18 trái dưới {bottom_left_18}")
    print(f"   Chiều dài: Ô 11 trái dưới {bottom_left_11} -> Ô 23 trái trên {top_left_23}")

    return (roi_x, roi_y, roi_width, roi_height)


def create_debug_image_direct(image: np.ndarray, detected_squares: List[Dict[str, Any]]) -> np.ndarray:
    """
    Tạo ảnh debug hiển thị các ô vuông đen đã detect được (phiên bản direct detection)

    Args:
        image: Ảnh gốc
        detected_squares: Danh sách các ô vuông đã detect

    Returns:
        Ảnh debug với các ô vuông được đánh dấu
    """
    debug_img = image.copy()

    for square in detected_squares:
        center = square["center"]
        bbox = square["bounding_box"]
        square_id = square["id"]

        # Vẽ hình chữ nhật bao quanh ô vuông
        x, y, w, h = bbox
        color = (0, 255, 0)  # Xanh lá cho tất cả ô vuông tìm thấy
        cv2.rectangle(debug_img, (x, y), (x + w, y + h), color, 2)

        # Vẽ điểm tâm
        cv2.circle(debug_img, center, 4, color, -1)

        # Vẽ số thứ tự
        cv2.putText(debug_img, str(square_id),
                   (center[0] - 10, center[1] - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Vẽ thông tin chi tiết (area)
        cv2.putText(debug_img, f"A:{square['area']:.0f}",
                   (center[0] - 15, center[1] + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Vẽ ROI Student ID
    student_id_roi = calculate_student_id_roi(detected_squares)
    if student_id_roi != (0, 0, 0, 0):
        x, y, w, h = student_id_roi
        # Vẽ hình chữ nhật ROI với màu đỏ
        cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 0, 255), 3)
        # Vẽ label cho ROI
        cv2.putText(debug_img, "Student ID ROI", (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Vẽ ROI Exam Code
    exam_code_roi = calculate_exam_code_roi(detected_squares)
    if exam_code_roi != (0, 0, 0, 0):
        x, y, w, h = exam_code_roi
        # Vẽ hình chữ nhật ROI với màu xanh dương
        cv2.rectangle(debug_img, (x, y), (x + w, y + h), (255, 0, 0), 3)
        # Vẽ label cho ROI
        cv2.putText(debug_img, "Exam Code ROI", (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Vẽ ROI Part 1
    part1_roi = calculate_part1_roi(detected_squares)
    if part1_roi != (0, 0, 0, 0):
        x, y, w, h = part1_roi
        # Vẽ hình chữ nhật ROI với màu tím
        cv2.rectangle(debug_img, (x, y), (x + w, y + h), (255, 0, 255), 3)
        # Vẽ label cho ROI
        cv2.putText(debug_img, "Part 1 ROI", (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

    # Vẽ ROI Part 2
    part2_roi = calculate_part2_roi(detected_squares)
    if part2_roi != (0, 0, 0, 0):
        x, y, w, h = part2_roi
        # Vẽ hình chữ nhật ROI với màu cam
        cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 165, 255), 3)
        # Vẽ label cho ROI
        cv2.putText(debug_img, "Part 2 ROI", (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

    # Vẽ ROI Part 3
    part3_roi = calculate_part3_roi(detected_squares)
    if part3_roi != (0, 0, 0, 0):
        x, y, w, h = part3_roi
        # Vẽ hình chữ nhật ROI với màu xanh lá
        cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        # Vẽ label cho ROI
        cv2.putText(debug_img, "Part 3 ROI", (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Vẽ thông tin tổng quan
    info_text = f"Found {len(detected_squares)} black squares"
    cv2.putText(debug_img, info_text, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    return debug_img


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




def detect_all_black_squares_direct(image: np.ndarray, debug: bool = False) -> List[Dict[str, Any]]:
    """
    Detect tất cả ô vuông đen trực tiếp từ ảnh sử dụng phương pháp tương tự omr_service

    Args:
        image: Ảnh đầu vào (BGR format)
        debug: Có in debug info không

    Returns:
        List các ô vuông đen được detect
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Tìm threshold tối ưu
    optimal_thresh, binary = find_optimal_threshold_for_squares(gray)

    # Tìm các contour
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_squares = []
    square_id = 1

    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Điều chỉnh ngưỡng area để cân bằng giữa ô lớn và ô nhỏ
        if area < 15 or area > 3000:  # Từ 20 pixels (4x5) đến 3000 pixels
            continue

        # Kiểm tra các điều kiện hình vuông - nới lỏng hơn để detect ô vuông nhỏ
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # Chấp nhận cả contour không hoàn hảo để detect ô vuông nhỏ
        if len(approx) >= 3:  # Chấp nhận từ 3 cạnh trở lên
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            bbox_area = w * h
            contour_area = cv2.contourArea(cnt)
            fill_ratio = contour_area / bbox_area if bbox_area > 0 else 0

            perimeter = cv2.arcLength(cnt, True)
            compactness = 4 * np.pi * contour_area / (perimeter * perimeter) if perimeter > 0 else 0

            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = contour_area / hull_area if hull_area > 0 else 0

            # Điều kiện để detect ô vuông đen, tránh hình tròn
            if (0.75 <= aspect_ratio <= 1.3 and  # Aspect ratio chặt cho hình vuông
                fill_ratio >= 0.75 and             # Fill ratio cao
                0.3 <= compactness <= 0.85 and    # Compactness của hình vuông (tránh hình tròn ~1.0)
                solidity >= 0.8):                # Solidity cao

                # Tính tọa độ tâm
                M = cv2.moments(cnt)
                if M['m00'] != 0:
                    center_x = int(M['m10'] / M['m00'])
                    center_y = int(M['m01'] / M['m00'])

                    square_info = {
                        "id": square_id,
                        "center": (center_x, center_y),
                        "bounding_box": (x, y, w, h),
                        "area": contour_area,
                        "found": True,
                        "aspect_ratio": aspect_ratio,
                        "fill_ratio": fill_ratio,
                        "compactness": compactness,
                        "solidity": solidity
                    }

                    detected_squares.append(square_info)
                    square_id += 1

                    if debug:
                        print(f"Square {square_id-1}: center=({center_x},{center_y}), area={contour_area:.1f}, "
                              f"aspect={aspect_ratio:.2f}, fill={fill_ratio:.2f}, "
                              f"compact={compactness:.2f}, solid={solidity:.2f}")

    # Sắp xếp theo tổng tọa độ x + y (từ góc trên-trái đến góc dưới-phải theo đường chéo)
    detected_squares.sort(key=lambda s: (s["center"][0] + s["center"][1]))

    # Cập nhật lại ID sau khi sắp xếp
    for i, square in enumerate(detected_squares):
        square["id"] = i + 1

    return detected_squares


def detect_all_black_squares(image_path: str, debug: bool = False, output_dir: str = "output") -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Detect tất cả ô vuông đen trong ảnh OMR sử dụng phương pháp hoàn toàn từ omr_service

    Args:
        image_path: Đường dẫn đến ảnh cần xử lý
        debug: Có tạo ảnh debug không
        output_dir: Thư mục lưu ảnh debug

    Returns:
        Tuple (results_dict, debug_image_path)
        - results_dict: Dictionary chứa thông tin về tất cả ô vuông đen
        - debug_image_path: Đường dẫn đến ảnh debug (nếu debug=True)

    Raises:
        ValueError: Nếu không thể đọc ảnh
    """
    # Đọc ảnh
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Không thể đọc ảnh: {image_path}")

    print(f"Kích thước ảnh gốc: {image.shape[1]}x{image.shape[0]}")

    # Sử dụng ảnh gốc không resize
    original_size = (image.shape[1], image.shape[0])
    resized = False
    print("Sử dụng ảnh gốc không resize")

    # Detect tất cả ô vuông đen trực tiếp
    detected_squares = detect_all_black_squares_direct(image, debug=debug)

    found_count = len(detected_squares)
    print(f"Tìm thấy {found_count} ô vuông đen")

    # Tạo kết quả trả về
    results = {
        "total_squares": found_count,
        "found_squares": found_count,
        "missing_squares": 0,
        "success_rate": 100.0 if found_count > 0 else 0.0,
        "image_info": {
            "original_size": original_size,
            "processed_size": (image.shape[1], image.shape[0]),
            "resized": resized
        },
        "squares": detected_squares
    }

    # Tạo ảnh debug nếu được yêu cầu
    debug_image_path = None
    if debug:
        # Tạo thư mục output nếu chưa tồn tại
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        debug_img = create_debug_image_direct(image, detected_squares)

        # Tạo tên file debug với timestamp
        timestamp = int(time.time())
        debug_filename = f"black_squares_debug_{timestamp}.png"
        debug_image_path = os.path.join(output_dir, debug_filename)

        # Lưu ảnh debug vào file
        cv2.imwrite(debug_image_path, debug_img)
        print(f"Đã lưu ảnh debug tại: {debug_image_path}")

    return results, debug_image_path
