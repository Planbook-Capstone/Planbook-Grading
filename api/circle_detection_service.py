import cv2
import numpy as np
import os
import base64
from .black_square_detection_service import (
    detect_all_black_squares_direct,
    calculate_student_id_roi,
    calculate_exam_code_roi,
    calculate_part1_roi,
    calculate_part2_roi,
    calculate_part3_roi
)


def _process_part3_circles(all_circles, filled_circles, part_number, debug_image):
    """
    Process Part 3 circles with the special grid structure:
    - 6 columns (questions/input fields)
    - 12 clusters per column: 1 minus, 2 comma positions, 10 digit rows (0-9) with 4 positions each
    """
    output_data = []
    student_answers = []
    filled_circles_set = set((x, y) for x, y, r in filled_circles)

    # Note: Part 3 filtering is now done during circle detection in _detect_circles_in_roi

    # Group circles by approximate column based on x coordinates
    columns = [[] for _ in range(6)]
    base = 200
    for x, y, r in all_circles:
        # Determine column based on x coordinate
        if x < base + 154:
            col_idx = 0
        elif x < base + 154 * 2 :
            col_idx = 1
        elif x < base + 154 * 3:
            col_idx = 2
        elif x < base + 154 * 4:
            col_idx = 3
        elif x < base + 154 * 5:
            col_idx = 4
        else:
            col_idx = 5

        columns[col_idx].append((x, y, r))

    # Process each column
    for col_idx, column_circles in enumerate(columns):
        if not column_circles:
            continue

        question_number = col_idx + 1

        # First, sort all circles by y coordinate to identify rows
        sorted_by_y = sorted(column_circles, key=lambda c: c[1])

        # Group circles into rows based on y coordinate proximity
        rows = []
        current_row = []

        for circle in sorted_by_y:
            if not current_row:
                current_row = [circle]
            elif abs(circle[1] - current_row[0][1]) <= 15:  # Same row threshold
                current_row.append(circle)
            else:
                if current_row:
                    # Sort circles within row by x coordinate
                    current_row.sort(key=lambda c: c[0])
                    rows.append(current_row)
                current_row = [circle]

        if current_row:
            current_row.sort(key=lambda c: c[0])
            rows.append(current_row)

        # Now we have rows sorted by y, and within each row, circles sorted by x
        # We expect 12 rows: 1 minus, 1 comma (with 2 circles), 10 digit rows (each with 4 circles)

        # Process the rows
        for row_idx, row_circles in enumerate(rows):
            if row_idx == 0:
                # First row: minus (should have 1 circle)
                for pos_idx, (x, y, r) in enumerate(row_circles):
                    symbol_label = f"{question_number}_minus_1"
                    label = f"part{part_number}_{symbol_label}_{x}_{y}"
                    output_data.append(label)

                    is_filled = (x, y) in filled_circles_set
                    if is_filled:
                        student_answers.append(label)

                    if debug_image is not None:
                        color = (0, 0, 255) if is_filled else (0, 255, 0)
                        cv2.circle(debug_image, (x, y), r, color, 2)
                        cv2.putText(debug_image, f"({symbol_label})", (x - 30, y - 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
                        cv2.putText(debug_image, symbol_label, (x - 15, y + 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

            elif row_idx == 1:
                # Second row: comma (should have 2 circles)
                for pos_idx, (x, y, r) in enumerate(row_circles):
                    comma_position = pos_idx + 2  # comma_2, comma_3
                    symbol_label = f"{question_number}_comma_{comma_position}"
                    label = f"part{part_number}_{symbol_label}_{x}_{y}"
                    output_data.append(label)

                    is_filled = (x, y) in filled_circles_set
                    if is_filled:
                        student_answers.append(label)

                    if debug_image is not None:
                        color = (0, 0, 255) if is_filled else (0, 255, 0)
                        cv2.circle(debug_image, (x, y), r, color, 2)
                        cv2.putText(debug_image, f"({symbol_label})", (x - 30, y - 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
                        cv2.putText(debug_image, symbol_label, (x - 15, y + 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

            else:
                # Rows 3-12: digits 0-9 (each should have 4 circles)
                digit = row_idx - 2  # 0, 1, 2, ..., 9
                for pos_idx, (x, y, r) in enumerate(row_circles):
                    x_position = pos_idx + 1  # 1, 2, 3, 4
                    symbol_label = f"{question_number}_{digit}_{x_position}"
                    label = f"part{part_number}_{symbol_label}_{x}_{y}"
                    output_data.append(label)

                    is_filled = (x, y) in filled_circles_set
                    if is_filled:
                        student_answers.append(label)

                    if debug_image is not None:
                        color = (0, 0, 255) if is_filled else (0, 255, 0)
                        cv2.circle(debug_image, (x, y), r, color, 2)
                        cv2.putText(debug_image, f"({symbol_label})", (x - 30, y - 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
                        cv2.putText(debug_image, symbol_label, (x - 15, y + 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    return output_data, student_answers

def _process_id_student_circles(all_circles, filled_circles, debug_image):
    """
    Process student ID circles and parse them into a student ID number
    Returns the student ID as a string of digits from left to right
    Cấu trúc: 6 cột (vị trí 1-6), mỗi cột có 10 hàng (số 0-9)
    Format: digit_position (ví dụ: 0_1, 1_2, 9_6)
    """
    output_data = []
    student_answers = []
    filled_circles_set = set((x, y) for x, y, r in filled_circles)

    # Group circles by approximate column (position 1-6) based on x coordinates
    columns = [[] for _ in range(6)]  # 6 cột cho student ID
    
    # Sort circles by x coordinate to determine columns
    sorted_by_x = sorted(all_circles, key=lambda c: c[0])
    
    if not sorted_by_x:
        return [], [], ""
    
    # Determine column boundaries based on x coordinates
    min_x = sorted_by_x[0][0]
    max_x = sorted_by_x[-1][0]
    
    # Tính toán chính xác để chia thành 6 cột
    total_width = max_x - min_x
    column_width = total_width / 6 if len(sorted_by_x) > 1 else 30
    
    for x, y, r in all_circles:
        # Determine which column (position 1-6) this circle belongs to
        col_idx = int((x - min_x) / column_width)
        # Đảm bảo cột cuối cùng không bị vượt quá index 5
        if col_idx >= 6:
            col_idx = 5
        col_idx = max(0, col_idx)  # Ensure within bounds (0-5 for 6 columns)
        columns[col_idx].append((x, y, r))

    student_id_digits = [""] * 6  # Initialize with empty strings for 6 positions

    # Collect all y coordinates for clustering
    all_y_coords = [y for x, y, r in all_circles]

    # Process each column (position)
    for col_idx, column_circles in enumerate(columns):
        if not column_circles:
            continue

        position = col_idx + 1  # Position 1-6

        # Sort by y coordinate to get rows (digits 0-9)
        sorted_circles = sorted(column_circles, key=lambda c: c[1])

        for x, y, r in sorted_circles:
            # Determine which digit (0-9) based on y coordinate using clustering
            digit = _get_digit_from_y_coordinate_id(y, all_y_coords)

            # Generate label for this circle: digit_position
            label = f"id_student_{digit}_{position}_{x}_{y}"
            output_data.append(label)

            is_filled = (x, y) in filled_circles_set
            if is_filled:
                student_answers.append(label)
                student_id_digits[col_idx] = str(digit)  # Store the filled digit

            if debug_image is not None:
                color = (0, 0, 255) if is_filled else (0, 255, 0)
                cv2.circle(debug_image, (x, y), r, color, 2)
                cv2.putText(debug_image, f"{digit}_{position}", (x - 15, y + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 255), 1)
    
    # Construct student ID from filled digits (left to right)
    student_id = "".join(student_id_digits).rstrip()  # Remove trailing empty strings

    return output_data, student_answers, student_id


def _get_digit_from_y_coordinate_id(y_coord, all_y_coords=None):
    """
    Determine which digit (0-9) based on y coordinate for student ID
    Uses clustering approach to group circles with similar y coordinates (±10 tolerance)
    and assigns digits 0-9 based on sorted cluster positions
    """
    if all_y_coords is None:
        # Fallback to old logic if no clustering data provided
        base_y = 260  # y coordinate của digit 0
        max_y = 560   # y coordinate của digit 9
        digit_spacing = (max_y - base_y) / 9
        digit = round((y_coord - base_y) / digit_spacing)
        return max(0, min(9, digit))

    # Clustering logic: group y coordinates with ±10 tolerance
    clusters = []
    tolerance = 10

    # Get unique y coordinates and sort them
    unique_y_coords = sorted(set(all_y_coords))

    # Group coordinates into clusters using tolerance
    for y in unique_y_coords:
        # Find if this y belongs to an existing cluster
        added_to_cluster = False
        for cluster in clusters:
            # Check if y is within tolerance of any point in the cluster
            if any(abs(y - cluster_y) <= tolerance for cluster_y in cluster):
                cluster.append(y)
                added_to_cluster = True
                break

        # If not added to any cluster, create new cluster
        if not added_to_cluster:
            clusters.append([y])

    # Calculate average y for each cluster and sort clusters by y
    cluster_data = []
    for cluster in clusters:
        avg_y = sum(cluster) / len(cluster)
        cluster_data.append((avg_y, cluster))

    # Sort clusters by average y coordinate (ascending)
    cluster_data.sort(key=lambda x: x[0])

    # Ensure we have exactly 10 clusters for digits 0-9
    if len(cluster_data) > 10:
        # If more than 10 clusters, keep only the first 10
        cluster_data = cluster_data[:10]
    elif len(cluster_data) < 10:
        # If less than 10 clusters, use interpolation for missing digits
        print(f"⚠️ Only found {len(cluster_data)} clusters, expected 10 for digits 0-9")

    # Find which cluster the current y_coord belongs to
    for digit, (avg_y, cluster) in enumerate(cluster_data):
        if any(abs(y_coord - cluster_y) <= tolerance for cluster_y in cluster):
            return min(digit, 9)  # Ensure digit is between 0-9

    # Fallback: if not found in any cluster, find closest cluster
    if cluster_data:
        closest_digit = 0
        min_distance = float('inf')
        for digit, (avg_y, cluster) in enumerate(cluster_data):
            distance = abs(y_coord - avg_y)
            if distance < min_distance:
                min_distance = distance
                closest_digit = digit
        return min(closest_digit, 9)

    return 0  # Default fallback




def _process_part1_circles(all_circles, filled_circles, part_number, starting_question_number, debug_image):
    output_data = []
    student_answers = []

    filled_circles_set = set((x, y) for x, y, r in filled_circles)

    sorted_by_x = sorted(all_circles, key=lambda c: c[0])

    main_blocks = []
    if sorted_by_x:
        current_block = [sorted_by_x[0]]
        block_threshold = 150
        for i in range(1, len(sorted_by_x)):
            avg_x_current_block = sum(c[0] for c in current_block) / len(current_block)
            if abs(sorted_by_x[i][0] - avg_x_current_block) < block_threshold:
                current_block.append(sorted_by_x[i])
            else:
                main_blocks.append(current_block)
                current_block = [sorted_by_x[i]]
        main_blocks.append(current_block)

    question_number = starting_question_number
    for block in main_blocks:
        sorted_block = sorted(block, key=lambda c: c[1])
        for i in range(0, len(sorted_block), 4):
            question_circles = sorted(sorted_block[i:i+4], key=lambda c: c[0])
            for j, (x, y, r) in enumerate(question_circles):
                answer_label = chr(ord('a') + j)
                label = f"part{part_number}_{question_number}_{answer_label}_{x}_{y}"
                output_data.append(label)

                is_filled = (x, y) in filled_circles_set
                if is_filled:
                    student_answers.append(label)

                if debug_image is not None:
                    color = (0, 0, 255) if is_filled else (0, 255, 0) # Red for filled, green for not filled
                    cv2.circle(debug_image, (x, y), r, color, 2)
                    cv2.putText(debug_image, f"{question_number}{answer_label}", (x - 15, y + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            question_number += 1
    return output_data, student_answers, question_number

def _process_part2_circles(all_circles, filled_circles, part_number, debug_image):
    output_data = []
    student_answers = []
    filled_circles_set = set((x, y) for x, y, r in filled_circles)

    # First, sort by Y to group into rows
    sorted_by_y = sorted(all_circles, key=lambda c: c[1])

    # Try different thresholds to find exactly 4 rows
    best_rows = None
    best_threshold = None

    # Try thresholds from 10 to 30 pixels
    for threshold in range(10, 31, 5):
        rows = []
        if sorted_by_y:
            current_row = [sorted_by_y[0]]
            for i in range(1, len(sorted_by_y)):
                if abs(sorted_by_y[i][1] - current_row[-1][1]) < threshold:
                    current_row.append(sorted_by_y[i])
                else:
                    rows.append(sorted(current_row, key=lambda c: c[0]))
                    current_row = [sorted_by_y[i]]
            rows.append(sorted(current_row, key=lambda c: c[0]))

        # Check if this threshold gives us exactly 4 rows
        if len(rows) == 4:
            best_rows = rows
            best_threshold = threshold
            print(f"✅ Found exactly 4 rows with threshold {threshold}")
            break
        else:
            print(f"🔍 Threshold {threshold}: found {len(rows)} rows")

    # If we couldn't find exactly 4 rows, return error
    if best_rows is None or len(best_rows) != 4:
        print(f"❌ Part2 error: Could not detect exactly 4 rows")
        print(f"   Tried thresholds 10-30, but couldn't group circles into 4 rows")

        # Log what we found for debugging
        threshold = 20  # Use default threshold for error reporting
        rows = []
        if sorted_by_y:
            current_row = [sorted_by_y[0]]
            for i in range(1, len(sorted_by_y)):
                if abs(sorted_by_y[i][1] - current_row[-1][1]) < threshold:
                    current_row.append(sorted_by_y[i])
                else:
                    rows.append(sorted(current_row, key=lambda c: c[0]))
                    current_row = [sorted_by_y[i]]
            rows.append(sorted(current_row, key=lambda c: c[0]))

        print(f"   With threshold {threshold}, detected {len(rows)} rows:")
        for i, row in enumerate(rows):
            print(f"     Row {i}: {len(row)} circles")
            if len(row) > 0:
                y_coords = [c[1] for c in row]
                print(f"       Y range: {min(y_coords)} - {max(y_coords)}")

        raise ValueError(f"Part 2: Không thể phát hiện đúng 4 hàng. Phát hiện {len(rows)} hàng thay vì 4 hàng. Vui lòng chụp lại ảnh rõ nét hơn.")

    rows = best_rows
    print(f"✅ Part2 detected exactly 4 rows with threshold {best_threshold}")

    # Log row details for debugging
    for i, row in enumerate(rows):
        print(f"  Row {i}: {len(row)} circles")
        if len(row) > 0:
            y_coords = [c[1] for c in row]
            print(f"    Y range: {min(y_coords)} - {max(y_coords)}")

    # Process all 4 rows
    # Each row should contain circles for 8 questions (16 circles total: 8 questions × 2 answers each)
    for q_idx in range(8):
        question_number = q_idx + 1
        # Each question has 2 circles per row (D and S)
        start_idx = q_idx * 2
        end_idx = start_idx + 2

        for r_idx, row in enumerate(rows):
            sub_part_label = chr(ord('a') + r_idx)  # a, b, c, d
            if end_idx <= len(row):
                pair = row[start_idx:end_idx]
                labels = ['D', 'S']
                for j, (x, y, r) in enumerate(pair):
                    answer_label = labels[j]
                    label = f"part{part_number}_{question_number}_{sub_part_label}_{answer_label}_{x}_{y}"
                    output_data.append(label)

                    is_filled = (x, y) in filled_circles_set
                    if is_filled:
                        student_answers.append(label)

                    if debug_image is not None:
                        color = (0, 0, 255) if is_filled else (0, 255, 0) # Red for filled, green for not filled
                        cv2.circle(debug_image, (x, y), r, color, 2)
                        cv2.putText(debug_image, f"{question_number}{sub_part_label}{answer_label}", (x - 15, y + 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

    print(f"✅ Part2 processed: {len(output_data)} total labels, {len(student_answers)} filled answers")
    return output_data, student_answers

def _detect_circles_in_roi(image, roi_coords, part_name=None, debug=False):
    """
    Detect circles in ROI with automatic param2 adjustment based on part type

    Args:
        image: Input image
        roi_coords: ROI coordinates (x_start, y_start, x_end, y_end)
        part_name: Part identifier to determine expected circle count
                  - 'student_id': 60 circles
                  - 'exam_code': 30 circles
                  - 'part1': 160 circles
                  - 'part2': 64 circles
                  - 'part3': 258 circles
        debug: Whether to save debug images

    Returns:
        tuple: (all_circles, filled_circles)

    Raises:
        ValueError: If circle count doesn't match expected count for the part
    """
    x_start, y_start, x_end, y_end = roi_coords

    # Validate ROI coordinates
    if image is None or len(image.shape) < 2:
        print(f"❌ Invalid image provided to _detect_circles_in_roi")
        return [], []

    img_height, img_width = image.shape[:2]

    # Ensure coordinates are within image boundaries
    x_start = max(0, min(x_start, img_width - 1))
    y_start = max(0, min(y_start, img_height - 1))
    x_end = max(x_start + 1, min(x_end, img_width))
    y_end = max(y_start + 1, min(y_end, img_height))

    # Extract ROI
    roi = image[y_start:y_end, x_start:x_end]

    # Validate ROI is not empty
    if roi.size == 0 or roi.shape[0] == 0 or roi.shape[1] == 0:
        print(f"❌ Empty ROI extracted: coords=({x_start},{y_start},{x_end},{y_end}), roi_shape={roi.shape}")
        return [], []

    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred_roi = cv2.medianBlur(gray_roi, 5)

    # Define expected circle counts for each part
    expected_counts = {
        'student_id': 60,
        'exam_code': 30,
        'part1': 160,
        'part2': 64,
        'part3': 258
    }

    expected_count = expected_counts.get(part_name, None)

    # Try different param2 values from 18 to 26 to find the right circle count
    best_circles = None
    best_filled_circles = None

    for param2_value in range(18, 27):
        circles = cv2.HoughCircles(
            blurred_roi, cv2.HOUGH_GRADIENT, dp=1, minDist=11,
            param1=95, param2=param2_value, minRadius=5, maxRadius=20
        )

        if circles is None:
            continue

        circles = np.round(circles[0, :]).astype("int")

        # Process circles to validate shape and detect filled ones
        all_circles = []
        filled_circles = []

        for (x, y, r) in circles:
            # Shape validation: Kiểm tra xem có phải thực sự là hình tròn không
            # Tạo contour từ circle để tính các metric
            mask = np.zeros(gray_roi.shape, dtype=np.uint8)
            cv2.circle(mask, (x, y), r, 255, -1)

            # Tìm contour của circle
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            contour = contours[0]
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)

            # Tính circularity = 4π * area / perimeter²
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
            else:
                circularity = 0

            # Tính aspect ratio của bounding rectangle
            _, _, w_rect, h_rect = cv2.boundingRect(contour)
            aspect_ratio = float(w_rect) / h_rect if h_rect > 0 else 0

            # Shape validation: Loại bỏ những shape không đủ tròn
            if circularity < 0.7 or aspect_ratio < 0.8 or aspect_ratio > 1.1:
                continue

            # Get the pixels within the circle from the original grayscale ROI
            circle_pixels = gray_roi[mask == 255]

            # Check if the circle is filled based on average pixel intensity
            if circle_pixels.size > 0:
                average_intensity = np.mean(circle_pixels)
                if average_intensity < 180:
                    filled_circles.append((x + x_start, y + y_start, r))

            all_circles.append((x + x_start, y + y_start, r))

        # Apply Part 3 specific filtering if needed
        if part_name == 'part3' and all_circles:
            # Filter out circles that are likely to be header text (like "C" letters)
            all_y_coords = [y for x, y, r in all_circles]
            min_y = min(all_y_coords)
            max_y = max(all_y_coords)

            # Filter out circles in the top 15% of the ROI (likely header area)
            y_range = max_y - min_y
            header_threshold = min_y + (y_range * 0.15)

            # Also filter by size - header text circles are often larger than answer circles
            filtered_all_circles = []
            filtered_filled_circles = []

            for x, y, r in all_circles:
                # Skip circles that are too high (in header area) or too large (text)
                if y > header_threshold and r <= 15:  # Only keep circles with radius <= 15
                    filtered_all_circles.append((x, y, r))
                    # Check if this circle was also in filled_circles
                    if any(abs(x - fx) < 2 and abs(y - fy) < 2 for fx, fy, _ in filled_circles):
                        # Find the corresponding filled circle
                        for fx, fy, fr in filled_circles:
                            if abs(x - fx) < 2 and abs(y - fy) < 2:
                                filtered_filled_circles.append((fx, fy, fr))
                                break

            original_count = len(all_circles)
            all_circles = filtered_all_circles
            filled_circles = filtered_filled_circles

            if original_count != len(all_circles):
                print(f"🔧 Part3 filtering during detection: {original_count} -> {len(all_circles)} circles (removed {original_count - len(all_circles)} header/text circles)")

        # Check if we found the expected number of circles
        if expected_count is not None and len(all_circles) == expected_count:
            print(f"✅ Found exact match for {part_name}: {len(all_circles)} circles with param2={param2_value}")
            best_circles = all_circles
            best_filled_circles = filled_circles
            break
        elif expected_count is None:
            # If no expected count specified, use the first successful detection
            best_circles = all_circles
            best_filled_circles = filled_circles
            break
        else:
            print(f"🔍 param2={param2_value}: found {len(all_circles)} circles (expected {expected_count})")

    # If we couldn't find the exact expected count, create debug image and raise error
    if expected_count is not None and (best_circles is None or len(best_circles) != expected_count):
        actual_count = len(best_circles) if best_circles else 0

        # Create debug image for error case
        if debug and part_name:
            debug_roi = roi.copy()

            # Draw all circles found (even if count is wrong)
            if best_circles:
                for (x, y, r) in best_circles:
                    # Convert back to ROI coordinates
                    roi_x = x - x_start
                    roi_y = y - y_start

                    # Check if filled
                    is_filled = best_filled_circles and any(abs(x - fx) < 2 and abs(y - fy) < 2 for fx, fy, _ in best_filled_circles)

                    # Draw circle
                    color = (0, 0, 255) if is_filled else (0, 255, 0)  # Red for filled, green for empty
                    cv2.circle(debug_roi, (roi_x, roi_y), r, color, 2)

                    # Add circle number
                    cv2.putText(debug_roi, str(len([c for c in best_circles if c[0] <= x and c[1] <= y])),
                               (roi_x - 10, roi_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

            # Add error title in red
            cv2.putText(debug_roi, f"ERROR: {part_name}: {actual_count}/{expected_count} circles",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            filled_count = len(best_filled_circles) if best_filled_circles else 0
            cv2.putText(debug_roi, f"Found {filled_count} filled circles",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Save debug image for error case
            debug_filename = f"debug_{part_name}_circles_ERROR.jpg"
            cv2.imwrite(debug_filename, debug_roi)
            print(f"💾 Error debug image saved: {debug_filename}")

        error_msg = f"Vui lòng chụp lại ảnh rõ nét, không chụp nghiêng hoặc ánh sáng không phù hợp. " \
                   f"Phần {part_name}: phát hiện {actual_count} ô, cần {expected_count} ô."
        print(f"❌ {error_msg}")
        raise ValueError(error_msg)

    # Log final results and create debug image
    if best_circles:
        filled_count = len(best_filled_circles)
        total_count = len(best_circles)
        print(f"✅ Final result for {part_name}: {total_count} total circles, {filled_count} filled circles")

        # Create debug image if requested
        if debug and part_name:
            debug_roi = roi.copy()

            # Draw all circles
            for (x, y, r) in best_circles:
                # Convert back to ROI coordinates
                roi_x = x - x_start
                roi_y = y - y_start

                # Check if filled
                is_filled = any(abs(x - fx) < 2 and abs(y - fy) < 2 for fx, fy, _ in best_filled_circles)

                # Draw circle
                color = (0, 0, 255) if is_filled else (0, 255, 0)  # Red for filled, green for empty
                cv2.circle(debug_roi, (roi_x, roi_y), r, color, 2)

                # Add circle number
                cv2.putText(debug_roi, str(len([c for c in best_circles if c[0] <= x and c[1] <= y])),
                           (roi_x - 10, roi_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

            # Add title
            cv2.putText(debug_roi, f"{part_name}: {total_count} circles ({filled_count} filled)",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Save debug image
            debug_filename = f"debug_{part_name}_circles.jpg"
            cv2.imwrite(debug_filename, debug_roi)
            print(f"💾 Debug image saved: {debug_filename}")

        # Log some circle details for debugging
        for i, (x, y, r) in enumerate(best_circles[:5]):  # Show first 5 circles
            is_filled = (x, y, r) in best_filled_circles or any(
                abs(x - fx) < 2 and abs(y - fy) < 2 for fx, fy, _ in best_filled_circles
            )
            print(f"  Circle {i+1}: ({x}, {y}), r={r}, filled={is_filled}")

    return best_circles or [], best_filled_circles or []

def _process_exam_code_circles(all_circles, filled_circles, debug_image):
    """
    Process circles for exam code detection (3 columns)
    Similar to student ID but with 3 columns instead of 6
    """
    output_data = []
    student_answers = []
    
    if not all_circles:
        return [], [], ""
    
    # Create set of filled circle coordinates for quick lookup
    filled_circles_set = set((x, y) for x, y, r in filled_circles)

    # Group circles by approximate column (position 1-3) based on x coordinates
    columns = [[] for _ in range(3)]  # 3 cột cho exam code
    
    # Sort circles by x coordinate to determine columns
    sorted_by_x = sorted(all_circles, key=lambda c: c[0])
    
    if not sorted_by_x:
        return [], [], ""
    
    # Determine column boundaries based on x coordinates
    min_x = sorted_by_x[0][0]
    max_x = sorted_by_x[-1][0]
    
    # Tính toán chính xác để chia thành 3 cột
    total_width = max_x - min_x

    # Xử lý trường hợp đặc biệt khi tất cả circles có cùng tọa độ x
    if total_width == 0 or len(sorted_by_x) <= 1:
        # Nếu chỉ có 1 circle hoặc tất cả có cùng x, đặt vào cột đầu tiên
        for x, y, r in all_circles:
            columns[0].append((x, y, r))
    else:
        column_width = total_width / 3

        for x, y, r in all_circles:
            # Determine which column (position 1-3) this circle belongs to
            col_idx = int((x - min_x) / column_width)
            # Đảm bảo cột cuối cùng không bị vượt quá index 2
            if col_idx >= 3:
                col_idx = 2
            col_idx = max(0, col_idx)  # Ensure within bounds (0-2 for 3 columns)
            columns[col_idx].append((x, y, r))

    exam_code_digits = [""] * 3  # Initialize with empty strings for 3 positions

    # Collect all y coordinates for clustering
    all_y_coords = [y for x, y, r in all_circles]

    # Process each column (position)
    for col_idx, column_circles in enumerate(columns):
        if not column_circles:
            continue

        position = col_idx + 1  # Position 1-3

        # Sort by y coordinate to get rows (digits 0-9)
        sorted_circles = sorted(column_circles, key=lambda c: c[1])

        for x, y, r in sorted_circles:
            # Determine which digit (0-9) based on y coordinate using clustering
            digit = _get_digit_from_y_coordinate_id(y, all_y_coords)

            # Generate label for this circle: digit_position
            label = f"exam_code_{digit}_{position}_{x}_{y}"
            output_data.append(label)

            is_filled = (x, y) in filled_circles_set
            if is_filled:
                student_answers.append(label)
                exam_code_digits[col_idx] = str(digit)  # Store the filled digit

            if debug_image is not None:
                color = (0, 0, 255) if is_filled else (0, 255, 0)
                cv2.circle(debug_image, (x, y), r, color, 2)
                cv2.putText(debug_image, f"{digit}_{position}", (x - 15, y + 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    # Join the digits to form the complete exam code
    exam_code = "".join(exam_code_digits)
    
    return output_data, student_answers, exam_code


def detect_circles(image_path, debug=False):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")

    # Print image dimensions for debugging
    img_height, img_width = image.shape[:2]
    print(f"🔍 Image dimensions: {img_width}x{img_height}")

    # Get ROIs from black square detection service
    print("🔍 Detecting black squares to determine ROIs...")
    detected_squares = detect_all_black_squares_direct(image)

    # Get ROI coordinates from black square detection
    student_id_roi = calculate_student_id_roi(detected_squares)
    exam_code_roi = calculate_exam_code_roi(detected_squares)
    part1_roi = calculate_part1_roi(detected_squares)
    part2_roi = calculate_part2_roi(detected_squares)
    part3_roi = calculate_part3_roi(detected_squares)

    print(f"📍 ROIs from black square detection:")
    print(f"   Student ID: {student_id_roi}")
    print(f"   Exam Code: {exam_code_roi}")
    print(f"   Part 1: {part1_roi}")
    print(f"   Part 2: {part2_roi}")
    print(f"   Part 3: {part3_roi}")

    # Convert ROI format from (x, y, width, height) to (x1, y1, x2, y2)
    def convert_roi_format(roi):
        if roi == (0, 0, 0, 0):
            return None
        x, y, w, h = roi
        return (x, y, x + w, y + h)

    # Convert ROI coordinates to (x1, y1, x2, y2) format
    roi_part_id_student = convert_roi_format(student_id_roi)
    roi_part_exam_code = convert_roi_format(exam_code_roi)
    roi_part1 = convert_roi_format(part1_roi)
    roi_part2 = convert_roi_format(part2_roi)
    roi_part3 = convert_roi_format(part3_roi)

    # Validate ROI coordinates
    rois_to_check = [
        ("Student ID", roi_part_id_student),
        ("Exam Code", roi_part_exam_code),
        ("Part 1", roi_part1),
        ("Part 2", roi_part2),
        ("Part 3", roi_part3)
    ]

    for roi_name, roi in rois_to_check:
        if roi is None:
            print(f"⚠️  {roi_name} ROI could not be determined from black squares")
        else:
            x1, y1, x2, y2 = roi
            if x2 > img_width or y2 > img_height or x1 < 0 or y1 < 0:
                print(f"⚠️  {roi_name} ROI ({x1},{y1},{x2},{y2}) extends beyond image bounds ({img_width}x{img_height})")
            else:
                print(f"✅ {roi_name} ROI ({x1},{y1},{x2},{y2}) is within image bounds")

    debug_image = image.copy() if debug else None
    all_answers = []
    student_answers = []
    student_id = ""
    exam_code = ""

    # Process Student ID
    if roi_part_id_student is not None:
        if debug:
            cv2.rectangle(debug_image, (roi_part_id_student[0], roi_part_id_student[1]), (roi_part_id_student[2], roi_part_id_student[3]), (255, 255, 0), 2)
            cv2.putText(debug_image, "Student ID ROI", (roi_part_id_student[0], roi_part_id_student[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        id_all_circles, id_filled_circles = _detect_circles_in_roi(image, roi_part_id_student, 'student_id', debug)
        id_data, id_student_answers, student_id = _process_id_student_circles(id_all_circles, id_filled_circles, debug_image)
        all_answers.extend(id_data)
        student_answers.extend(id_student_answers)
    else:
        print("⚠️ Skipping Student ID processing - ROI not available")

    # Process Exam Code
    if roi_part_exam_code is not None:
        if debug:
            cv2.rectangle(debug_image, (roi_part_exam_code[0], roi_part_exam_code[1]), (roi_part_exam_code[2], roi_part_exam_code[3]), (255, 0, 255), 2)
            cv2.putText(debug_image, "Exam Code ROI", (roi_part_exam_code[0], roi_part_exam_code[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        exam_all_circles, exam_filled_circles = _detect_circles_in_roi(image, roi_part_exam_code, 'exam_code', debug)
        exam_data, exam_student_answers, exam_code = _process_exam_code_circles(exam_all_circles, exam_filled_circles, debug_image)
        all_answers.extend(exam_data)
        student_answers.extend(exam_student_answers)
    else:
        print("⚠️ Skipping Exam Code processing - ROI not available")

    # Process Part 1
    if roi_part1 is not None:
        if debug:
            cv2.rectangle(debug_image, (roi_part1[0], roi_part1[1]), (roi_part1[2], roi_part1[3]), (255, 0, 0), 2)
            cv2.putText(debug_image, "Part 1 ROI", (roi_part1[0], roi_part1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        part1_all_circles, part1_filled_circles = _detect_circles_in_roi(image, roi_part1, 'part1', debug)
        part1_data, part1_student, _ = _process_part1_circles(part1_all_circles, part1_filled_circles, 1, 1, debug_image)
        all_answers.extend(part1_data)
        student_answers.extend(part1_student)
    else:
        print("⚠️ Skipping Part 1 processing - ROI not available")

    # Process Part 2
    if roi_part2 is not None:
        if debug:
            cv2.rectangle(debug_image, (roi_part2[0], roi_part2[1]), (roi_part2[2], roi_part2[3]), (0, 0, 255), 2)
            cv2.putText(debug_image, "Part 2 ROI", (roi_part2[0], roi_part2[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        part2_all_circles, part2_filled_circles = _detect_circles_in_roi(image, roi_part2, 'part2', debug)
        part2_data, part2_student = _process_part2_circles(part2_all_circles, part2_filled_circles, 2, debug_image)
        all_answers.extend(part2_data)
        student_answers.extend(part2_student)
    else:
        print("⚠️ Skipping Part 2 processing - ROI not available")

    # Process Part 3
    if roi_part3 is not None:
        if debug:
            cv2.rectangle(debug_image, (roi_part3[0], roi_part3[1]), (roi_part3[2], roi_part3[3]), (0, 255, 255), 2)
            cv2.putText(debug_image, "Part 3 ROI", (roi_part3[0], roi_part3[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        part3_all_circles, part3_filled_circles = _detect_circles_in_roi(image, roi_part3, 'part3', debug)
        part3_data, part3_student = _process_part3_circles(part3_all_circles, part3_filled_circles, 3, debug_image)
        all_answers.extend(part3_data)
        student_answers.extend(part3_student)
    else:
        print("⚠️ Skipping Part 3 processing - ROI not available")

    # Tạo các ảnh debug
    debug_images = {}
    debug_image_path = None

    if debug and debug_image is not None:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_dir = "debug_circles"
        os.makedirs(output_dir, exist_ok=True)
        debug_image_path = os.path.join(output_dir, f"{base_name}_circles_debug.png")
        cv2.imwrite(debug_image_path, debug_image)

    
        # Chuyển thành base64 để trả về trong API
        debug_images = {
            "original_debug": image_to_base64(debug_image)
        }

        print(f"Debug images saved:")

        print(f"  - Original debug: {debug_image_path}")

    # Format student answers into structured format
    formatted_answers = _format_student_answers(student_answers)

    return {
        "all_answers": all_answers,
        "student_answers": student_answers,
        "student_id": student_id,
        "exam_code": exam_code,
        "formatted_answers": formatted_answers,
        "debug_images": debug_images
    }, debug_image_path


def _format_student_answers(student_answers):
    """
    Format raw student answers into structured format by parts
    """
    formatted = {
        "part1": {},
        "part2": {},
        "part3": {}
    }

    for answer in student_answers:
        if answer.startswith("part1_"):
            # Format: part1_1_a_236_702 -> question 1, answer a
            parts = answer.split("_")
            if len(parts) >= 3:
                question_num = parts[1]
                answer_choice = parts[2].upper()
                formatted["part1"][question_num] = answer_choice

        elif answer.startswith("part2_"):
            # Format: part2_1_a_D_238_1054 -> question 1, sub_part a, answer D
            parts = answer.split("_")
            if len(parts) >= 4:
                question_num = parts[1]
                sub_part = parts[2]
                answer_choice = parts[3]

                if question_num not in formatted["part2"]:
                    formatted["part2"][question_num] = {}
                formatted["part2"][question_num][sub_part] = answer_choice

        elif answer.startswith("part3_"):
            # Format: part3_1_minus_1_232_1270 or part3_1_comma_3_298_1294 or part3_1_2_4_330_1370
            parts = answer.split("_")
            if len(parts) >= 4:
                question_num = parts[1]
                symbol_or_digit = parts[2]
                position = int(parts[3]) if parts[3].isdigit() else 0

                if question_num not in formatted["part3"]:
                    # Initialize with empty positions (4 positions: 1,2,3,4)
                    formatted["part3"][question_num] = ["", "", "", ""]

                # Convert symbol to actual character
                if symbol_or_digit == "minus":
                    char = "-"
                elif symbol_or_digit == "comma":
                    char = ","
                else:
                    # It's a digit
                    char = symbol_or_digit

                # Place character at correct position (1-based to 0-based)
                if 1 <= position <= 4:
                    formatted["part3"][question_num][position - 1] = char

    # Convert part3 arrays to strings
    for question_num, positions in formatted["part3"].items():
        if isinstance(positions, list):
            # Join positions and remove trailing empty positions
            answer_str = "".join(positions).rstrip()
            formatted["part3"][question_num] = answer_str

    return formatted


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
