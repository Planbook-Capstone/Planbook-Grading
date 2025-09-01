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
    """
    output_data = []
    student_answers = []
    filled_circles_set = set((x, y) for x, y, r in filled_circles)

    # Group circles by approximate column (digit position) based on x coordinates
    columns = [[] for _ in range(8)]  # Assuming max 8 digits for student ID
    
    # Sort circles by x coordinate to determine columns
    sorted_by_x = sorted(all_circles, key=lambda c: c[0])
    
    if not sorted_by_x:
        return [], [], ""
    
    # Determine column boundaries based on x coordinates
    min_x = sorted_by_x[0][0]
    max_x = sorted_by_x[-1][0]
    column_width = (max_x - min_x) / 7 if len(sorted_by_x) > 1 else 50  # Divide into columns
    
    for x, y, r in all_circles:
        # Determine which column (digit position) this circle belongs to
        col_idx = int((x - min_x) / column_width)
        col_idx = max(0, min(7, col_idx))  # Ensure within bounds
        columns[col_idx].append((x, y, r))

    student_id_digits = [""] * 8  # Initialize with empty strings
    
    # Process each column (digit position)
    for col_idx, column_circles in enumerate(columns):
        if not column_circles:
            continue
            
        # Sort by y coordinate to get rows (digits 0-9)
        sorted_circles = sorted(column_circles, key=lambda c: c[1])
        
        for x, y, r in sorted_circles:
            # Determine which digit (0-9) based on y coordinate
            # Assuming digits are arranged vertically from 0 to 9
            digit = _get_digit_from_y_coordinate(y)
            
            # Generate label for this circle
            label = f"id_student_{col_idx+1}_{digit}_{x}_{y}"
            output_data.append(label)
            
            is_filled = (x, y) in filled_circles_set
            if is_filled:
                student_answers.append(label)
                student_id_digits[col_idx] = str(digit)  # Store the filled digit
            
            if debug_image is not None:
                color = (0, 0, 255) if is_filled else (0, 255, 0)
                cv2.circle(debug_image, (x, y), r, color, 2)
                cv2.putText(debug_image, f"ID{col_idx+1}_{digit}", (x - 15, y + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 255), 1)
    
    # Construct student ID from filled digits (left to right)
    student_id = "".join(student_id_digits).rstrip()  # Remove trailing empty strings
    
    return output_data, student_answers, student_id


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
    if len(rows) != 4:
        # Fallback or error for unexpected row count
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

def detect_circles(image_path, debug=False):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")

    # Define ROIs
    roi_part_id_student = (1050, 80, 750, 230)  # ROI for student ID section
    roi_part1 = (200, 680, 1200, 950)
    roi_part2 = (200, 990, 1200, 1163)
    roi_part3 = (200, 1220, 1110, 1558) # Adjusted coordinates for Part 3

    debug_image = image.copy() if debug else None
    all_answers = []
    student_answers = []
    student_id = ""

    # Process Student ID
    if debug:
        cv2.rectangle(debug_image, (roi_part_id_student[0], roi_part_id_student[1]), (roi_part_id_student[2], roi_part_id_student[3]), (255, 255, 0), 2)
        cv2.putText(debug_image, "Student ID ROI", (roi_part_id_student[0], roi_part_id_student[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    id_all_circles, id_filled_circles = _detect_circles_in_roi(image, roi_part_id_student)
    id_data, id_student_answers, student_id = _process_id_student_circles(id_all_circles, id_filled_circles, debug_image)
    all_answers.extend(id_data)
    student_answers.extend(id_student_answers)

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

    return {"all_answers": all_answers, "student_answers": student_answers, "student_id": student_id}, debug_image_path
