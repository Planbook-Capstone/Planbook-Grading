import cv2
import numpy as np
import os


def _process_part3_circles(all_circles, filled_circles, part_number, debug_image):
    """
    Process Part 3 circles with the special grid structure:
    - 6 columns (questions/input fields)
    - Multiple rows: minus sign, comma positions, digits 0-9
    - 4 positions per row (a,b,c,d)
    """
    output_data = []
    student_answers = []
    filled_circles_set = set((x, y) for x, y, r in filled_circles)

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

        # Sort by y coordinate to get rows
        sorted_circles = sorted(column_circles, key=lambda c: c[1])

        # Group into rows based on y coordinate proximity
        rows = []
        current_row = []

        for circle in sorted_circles:
            if not current_row:
                current_row = [circle]
            elif abs(circle[1] - current_row[0][1]) < 15:  # Same row threshold
                current_row.append(circle)
            else:
                if current_row:
                    rows.append(sorted(current_row, key=lambda c: c[0]))  # Sort by x within row
                current_row = [circle]

        if current_row:
            rows.append(sorted(current_row, key=lambda c: c[0]))

        # Process each row
        for row_idx, row_circles in enumerate(rows):
            for pos_idx, (x, y, r) in enumerate(row_circles):
                # Determine the symbol/digit label based on y coordinate and position
                symbol_label = _get_part3_symbol_label(pos_idx, y)

                # Generate label in the new format: part3_question_symbol_x_y
                label = f"part{part_number}_{question_number}_{symbol_label}_{x}_{y}"
                output_data.append(label)

                is_filled = (x, y) in filled_circles_set
                if is_filled:
                    student_answers.append(label)

                # Generate marking text for debug display
                marking_text = f"({question_number}_{symbol_label})"

                if debug_image is not None:
                    color = (0, 0, 255) if is_filled else (0, 255, 0)
                    cv2.circle(debug_image, (x, y), r, color, 2)
                    cv2.putText(debug_image, marking_text, (x - 30, y - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
                    cv2.putText(debug_image, f"{question_number}_{symbol_label}", (x - 15, y + 5),
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
    
    # Process each column (position)
    for col_idx, column_circles in enumerate(columns):
        if not column_circles:
            continue
            
        position = col_idx + 1  # Position 1-6
        
        # Sort by y coordinate to get rows (digits 0-9)
        sorted_circles = sorted(column_circles, key=lambda c: c[1])
        
        for x, y, r in sorted_circles:
            # Determine which digit (0-9) based on y coordinate
            digit = _get_digit_from_y_coordinate_id(y)
            
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
    student_id = "".join(student_id_digits).rstrip()  # Remove trailing empty strings
    
    return output_data, student_answers, student_id


def _get_digit_from_y_coordinate_id(y_coord):
    """
    Determine which digit (0-9) based on y coordinate for student ID
    Assumes digits are arranged vertically in order 0,1,2,3,4,5,6,7,8,9
    """
    # Dựa trên dữ liệu thực tế: y từ 260 đến 546
    # Digit 0 ở y ≈ 260, digit 9 ở y ≈ 546
    base_y = 260  # y coordinate của digit 0
    max_y = 546   # y coordinate của digit 9
    
    # Tính spacing cho 10 digits (0-9)
    digit_spacing = (max_y - base_y) / 9  # Chia khoảng cách cho 9 khoảng (0->1, 1->2, ..., 8->9)
    
    # Tính digit dựa trên vị trí y
    digit = round((y_coord - base_y) / digit_spacing)
    return max(0, min(9, digit))  # Ensure digit is between 0-9


def _get_digit_from_y_coordinate(y_coord):
    """
    Determine which digit (0-9) based on y coordinate
    Assumes digits are arranged vertically in order 0,1,2,3,4,5,6,7,8,9
    """
    # These values may need adjustment based on actual image layout
    base_y = 100  # Approximate y coordinate of digit 0
    digit_spacing = 24  # Approximate spacing between digits
    
    digit = int((y_coord - base_y) / digit_spacing)
    return max(0, min(9, digit))  # Ensure digit is between 0-9


def _get_part3_symbol_label(pos_idx, y_coord):
    """
    Generate the symbol/digit label for Part 3 based on position and y coordinate
    Returns labels like: minus, comma_1, comma_2, 0_1, 0_2, 1_1, etc.
    """
    # Determine symbol/digit based on y coordinate
    if y_coord < 1280:  # Minus row
        if pos_idx == 0:  # Only first position has minus
            return f"minus_{pos_idx+1}"

    elif y_coord < 1310:  # Comma row
        if pos_idx in [0, 1]:  # Middle positions have comma
            return f"comma_{pos_idx+2}"
        
    else:  # Digit rows (0-9)
        # Calculate which digit based on y coordinate
        digit = max(0, min(9, (y_coord - 1315) // 24))
        position = pos_idx + 1  # Convert to 1-4
        return f"{digit}_{position}"
    """
    Generate the symbol/digit label for Part 3 based on position and y coordinate
    Returns labels like: minus, comma_1, comma_2, 0_1, 0_2, 1_1, etc.
    """
    # Determine symbol/digit based on y coordinate
    if y_coord < 1280:  # Minus row
        if pos_idx == 0:  # Only first position has minus
            return f"minus_{pos_idx+1}"

    elif y_coord < 1310:  # Comma row
        if pos_idx in [0, 1]:  # Middle positions have comma
            return f"comma_{pos_idx+2}"
        
    else:  # Digit rows (0-9)
        # Calculate which digit based on y coordinate
        digit = max(0, min(9, (y_coord - 1315) // 24))
        position = pos_idx + 1  # Convert to 1-4
        return f"{digit}_{position}"

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

    rows = []
    if sorted_by_y:
        current_row = [sorted_by_y[0]]
        for i in range(1, len(sorted_by_y)):
            if abs(sorted_by_y[i][1] - current_row[-1][1]) < 20: # y-threshold
                current_row.append(sorted_by_y[i])
            else:
                rows.append(sorted(current_row, key=lambda c: c[0]))
                current_row = [sorted_by_y[i]]
        rows.append(sorted(current_row, key=lambda c: c[0]))

    # There should be 4 rows (a, b, c, d)
    print(f"DEBUG: Part2 detected {len(rows)} rows, expected 4")
    if len(rows) != 4:
        # Fallback or error for unexpected row count
        print(f"DEBUG: Part2 row count mismatch - detected {len(rows)} rows instead of 4")
        for i, row in enumerate(rows):
            print(f"  Row {i}: {len(row)} circles")
        return [], []

    # Each row contains circles for 8 questions
    for q_idx in range(8):
        question_number = q_idx + 1
        # Each question has 2 circles per row (D and S)
        start_idx = q_idx * 2
        end_idx = start_idx + 2

        for r_idx, row in enumerate(rows):
            sub_part_label = chr(ord('a') + r_idx)
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
    return output_data, student_answers

def _detect_circles_in_roi(image, roi_coords):
    x_start, y_start, x_end, y_end = roi_coords
    roi = image[y_start:y_end, x_start:x_end]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred_roi = cv2.medianBlur(gray_roi, 5)

    circles = cv2.HoughCircles(
        blurred_roi, cv2.HOUGH_GRADIENT, dp=1, minDist=18,
        param1=80, param2=20, minRadius=5, maxRadius=20
    )

    if circles is None:
        return [], []

    circles = np.round(circles[0, :]).astype("int")

    all_circles = []
    filled_circles = []

    for (x, y, r) in circles:
        # Create a mask for the circle
        mask = np.zeros(gray_roi.shape, dtype=np.uint8)
        cv2.circle(mask, (x, y), r, 255, -1)

        # Get the pixels within the circle from the original grayscale ROI
        circle_pixels = gray_roi[mask == 255]

        # Check if the circle is filled based on average pixel intensity
        if circle_pixels.size > 0:
            average_intensity = np.mean(circle_pixels)
            print(f"Circle at ({x + x_start}, {y + y_start}), r={r}, avg_intensity={average_intensity:.2f}")
            if average_intensity < 180:
                filled_circles.append((x + x_start, y + y_start, r))

        all_circles.append((x + x_start, y + y_start, r))

    return all_circles, filled_circles

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
    column_width = total_width / 3 if len(sorted_by_x) > 1 else 30
    
    for x, y, r in all_circles:
        # Determine which column (position 1-3) this circle belongs to
        col_idx = int((x - min_x) / column_width)
        # Đảm bảo cột cuối cùng không bị vượt quá index 2
        if col_idx >= 3:
            col_idx = 2
        col_idx = max(0, col_idx)  # Ensure within bounds (0-2 for 3 columns)
        columns[col_idx].append((x, y, r))

    exam_code_digits = [""] * 3  # Initialize with empty strings for 3 positions
    
    # Process each column (position)
    for col_idx, column_circles in enumerate(columns):
        if not column_circles:
            continue
            
        position = col_idx + 1  # Position 1-3
        
        # Sort by y coordinate to get rows (digits 0-9)
        sorted_circles = sorted(column_circles, key=lambda c: c[1])
        
        for x, y, r in sorted_circles:
            # Determine which digit (0-9) based on y coordinate
            digit = _get_digit_from_y_coordinate_id(y)
            
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

    # Define ROIs
    roi_part_id_student = (910, 200 , 1050, 600)  # ROI for student ID section
    roi_part_exam_code = (1080, 200, 1150, 600)     # ROI for exam code section (3 columns)
    roi_part1 = (200, 680, 1200, 950)
    roi_part2 = (200, 990, 1200, 1163)
    roi_part3 = (200, 1220, 1110, 1558) # Adjusted coordinates for Part 3

    debug_image = image.copy() if debug else None
    all_answers = []
    student_answers = []
    student_id = ""
    exam_code = ""

    # Process Student ID
    if debug:
        cv2.rectangle(debug_image, (roi_part_id_student[0], roi_part_id_student[1]), (roi_part_id_student[2], roi_part_id_student[3]), (255, 255, 0), 2)
        cv2.putText(debug_image, "Student ID ROI", (roi_part_id_student[0], roi_part_id_student[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    id_all_circles, id_filled_circles = _detect_circles_in_roi(image, roi_part_id_student)
    id_data, id_student_answers, student_id = _process_id_student_circles(id_all_circles, id_filled_circles, debug_image)
    all_answers.extend(id_data)
    student_answers.extend(id_student_answers)

    # Process Exam Code
    if debug:
        cv2.rectangle(debug_image, (roi_part_exam_code[0], roi_part_exam_code[1]), (roi_part_exam_code[2], roi_part_exam_code[3]), (255, 0, 255), 2)
        cv2.putText(debug_image, "Exam Code ROI", (roi_part_exam_code[0], roi_part_exam_code[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

    exam_all_circles, exam_filled_circles = _detect_circles_in_roi(image, roi_part_exam_code)
    exam_data, exam_student_answers, exam_code = _process_exam_code_circles(exam_all_circles, exam_filled_circles, debug_image)
    all_answers.extend(exam_data)
    student_answers.extend(exam_student_answers)

    # Process Part 1
    if debug:
        cv2.rectangle(debug_image, (roi_part1[0], roi_part1[1]), (roi_part1[2], roi_part1[3]), (255, 0, 0), 2)
        cv2.putText(debug_image, "Part 1 ROI", (roi_part1[0], roi_part1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    part1_all_circles, part1_filled_circles = _detect_circles_in_roi(image, roi_part1)
    part1_data, part1_student, _ = _process_part1_circles(part1_all_circles, part1_filled_circles, 1, 1, debug_image)
    all_answers.extend(part1_data)
    student_answers.extend(part1_student)

    # Process Part 2
    if debug:
        cv2.rectangle(debug_image, (roi_part2[0], roi_part2[1]), (roi_part2[2], roi_part2[3]), (0, 0, 255), 2)
        cv2.putText(debug_image, "Part 2 ROI", (roi_part2[0], roi_part2[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    part2_all_circles, part2_filled_circles = _detect_circles_in_roi(image, roi_part2)
    part2_data, part2_student = _process_part2_circles(part2_all_circles, part2_filled_circles, 2, debug_image)
    all_answers.extend(part2_data)
    student_answers.extend(part2_student)

    # Process Part 3
    if debug:
        cv2.rectangle(debug_image, (roi_part3[0], roi_part3[1]), (roi_part3[2], roi_part3[3]), (0, 255, 255), 2)
        cv2.putText(debug_image, "Part 3 ROI", (roi_part3[0], roi_part3[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    part3_all_circles, part3_filled_circles = _detect_circles_in_roi(image, roi_part3)
    part3_data, part3_student = _process_part3_circles(part3_all_circles, part3_filled_circles, 3, debug_image)
    all_answers.extend(part3_data)
    student_answers.extend(part3_student)

    debug_image_path = None
    if debug and debug_image is not None:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_dir = "debug_circles"
        os.makedirs(output_dir, exist_ok=True)
        debug_image_path = os.path.join(output_dir, f"{base_name}_circles_debug.png")
        cv2.imwrite(debug_image_path, debug_image)

    # Format student answers into structured format
    formatted_answers = _format_student_answers(student_answers)

    return {
        "all_answers": all_answers,
        "student_answers": student_answers,
        "student_id": student_id,
        "exam_code": exam_code,
        "formatted_answers": formatted_answers
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
