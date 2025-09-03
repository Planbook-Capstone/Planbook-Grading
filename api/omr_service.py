
import cv2
import numpy as np
from PIL import Image
import os
import math
import time


def calculate_distance(p1, p2):
    """Tính khoảng cách giữa 2 điểm"""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def select_four_extreme_markers_safe(square_markers):
    """
    Chọn 4 marker extreme một cách an toàn, đảm bảo không trùng lặp

    Args:
        square_markers: List of (cx, cy, corners_sorted) tuples

    Returns:
        tuple: (tl_idx, tr_idx, bl_idx, br_idx) - indices của 4 marker được chọn

    Raises:
        ValueError: Nếu không đủ 4 marker hoặc không thể chọn 4 marker khác nhau
    """
    if len(square_markers) < 4:
        raise ValueError(f"Không đủ marker để chọn 4 đỉnh. Chỉ có {len(square_markers)} marker.")

    centers = np.array([(cx, cy) for cx, cy, _ in square_markers])
    used_indices = set()

    # Bước 1: Chọn top-left marker (x+y nhỏ nhất)
    available_indices = [i for i in range(len(centers)) if i not in used_indices]
    tl_idx = min(available_indices, key=lambda i: centers[i][0] + centers[i][1])
    used_indices.add(tl_idx)
    print(f"🔍 Selected top-left marker: index={tl_idx}, center={centers[tl_idx]}")

    # Bước 2: Chọn top-right marker (x-y lớn nhất trong các marker còn lại)
    available_indices = [i for i in range(len(centers)) if i not in used_indices]
    if not available_indices:
        raise ValueError("Không thể chọn top-right marker - không còn marker nào khả dụng")
    tr_idx = max(available_indices, key=lambda i: centers[i][0] - centers[i][1])
    used_indices.add(tr_idx)
    print(f"🔍 Selected top-right marker: index={tr_idx}, center={centers[tr_idx]}")

    # Bước 3: Chọn bottom-left marker (y-x lớn nhất trong các marker còn lại)
    available_indices = [i for i in range(len(centers)) if i not in used_indices]
    if not available_indices:
        raise ValueError("Không thể chọn bottom-left marker - không còn marker nào khả dụng")
    bl_idx = max(available_indices, key=lambda i: centers[i][1] - centers[i][0])
    used_indices.add(bl_idx)
    print(f"🔍 Selected bottom-left marker: index={bl_idx}, center={centers[bl_idx]}")

    # Bước 4: Chọn bottom-right marker (x+y lớn nhất trong các marker còn lại)
    available_indices = [i for i in range(len(centers)) if i not in used_indices]
    if not available_indices:
        raise ValueError("Không thể chọn bottom-right marker - không còn marker nào khả dụng")
    br_idx = max(available_indices, key=lambda i: centers[i][0] + centers[i][1])
    used_indices.add(br_idx)
    print(f"🔍 Selected bottom-right marker: index={br_idx}, center={centers[br_idx]}")

    # Validation: Đảm bảo 4 indices khác nhau
    selected_indices = {tl_idx, tr_idx, bl_idx, br_idx}
    if len(selected_indices) != 4:
        raise ValueError(f"Lỗi logic: Đã chọn trùng marker. Indices: tl={tl_idx}, tr={tr_idx}, bl={bl_idx}, br={br_idx}")

    print(f"✅ Successfully selected 4 unique markers: {selected_indices}")
    return tl_idx, tr_idx, bl_idx, br_idx

def validate_marker_selection(centers, tl_idx, tr_idx, bl_idx, br_idx, min_distance=50):
    """
    Validate rằng 4 marker được chọn hợp lý

    Args:
        centers: Array of marker centers
        tl_idx, tr_idx, bl_idx, br_idx: Indices của 4 marker được chọn
        min_distance: Khoảng cách tối thiểu giữa các marker

    Raises:
        ValueError: Nếu validation fail
    """
    indices = [tl_idx, tr_idx, bl_idx, br_idx]

    # Kiểm tra khoảng cách tối thiểu giữa các marker
    for i in range(len(indices)):
        for j in range(i+1, len(indices)):
            dist = np.linalg.norm(centers[indices[i]] - centers[indices[j]])
            if dist < min_distance:
                raise ValueError(f"Marker {indices[i]} và {indices[j]} quá gần nhau (distance={dist:.1f} < {min_distance})")

    print(f"✅ Marker validation passed - all markers are sufficiently spaced (min_distance={min_distance})")
    return True



def debug_marker_detection(image, thresh, contours, image_path):
    """Debug chi tiết quá trình phát hiện marker"""
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    debug_dir = f"debug_marker_detection"
    os.makedirs(debug_dir, exist_ok=True)

    # Lưu ảnh threshold
    cv2.imwrite(os.path.join(debug_dir, f"{base_name}_threshold.png"), thresh)

    # Tạo ảnh debug với tất cả contours
    debug_img = image.copy()
    debug_info = []

    print(f"Total contours found: {len(contours)}")

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < 200:  # Bỏ qua contour quá nhỏ
            continue

        # Vẽ tất cả contours có area > 100
        cv2.drawContours(debug_img, [cnt], -1, (128, 128, 128), 1)  # Màu xám cho tất cả

        # Approximation cho tất cả contours
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # Tính các metrics cho tất cả
        _, _, w, h = cv2.boundingRect(approx)
        aspect_ratio = float(w) / h
        bbox_area = w * h
        contour_area = cv2.contourArea(cnt)
        fill_ratio = contour_area / bbox_area if bbox_area > 0 else 0
        perimeter = cv2.arcLength(cnt, True)
        compactness = 4 * np.pi * contour_area / (perimeter * perimeter) if perimeter > 0 else 0

        # Tính solidity
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = contour_area / hull_area if hull_area > 0 else 0

        # Kiểm tra điều kiện
        is_4_vertices = len(approx) == 4
        is_convex = cv2.isContourConvex(approx)
        good_aspect = 0.85 < aspect_ratio < 1.1
        good_fill = fill_ratio > 0.7
        good_compact = 0.4 < compactness < 0.85  # Không quá tròn
        good_solidity = solidity > 0.85
        good_area = 1000 <= area <= 3200

        is_valid_marker = (is_4_vertices and is_convex and
                          good_aspect and good_fill and good_compact and
                          good_solidity and good_area)

        # Vẽ và ghi thông tin cho contours có area phù hợp
        if area >= 500:  # Chỉ hiển thị contours lớn hơn
            M = cv2.moments(approx)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])

                # Màu sắc dựa trên tính hợp lệ
                if is_valid_marker:
                    color = (0, 255, 0)  # Xanh lá - valid
                elif good_area and is_4_vertices:
                    color = (0, 255, 255)  # Vàng - gần đúng
                else:
                    color = (0, 0, 255)  # Đỏ - không hợp lệ

                cv2.circle(debug_img, (cx, cy), 8, color, -1)
                cv2.putText(debug_img, str(i), (cx-10, cy-15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                debug_info.append({
                    'id': i,
                    'center': (cx, cy),
                    'area': contour_area,
                    'vertices': len(approx),
                    'convex': is_convex,
                    'aspect_ratio': aspect_ratio,
                    'fill_ratio': fill_ratio,
                    'compactness': compactness,
                    'solidity': solidity,
                    'good_area': good_area,
                    'is_valid': is_valid_marker
                })

    # Lưu ảnh debug
    cv2.imwrite(os.path.join(debug_dir, f"{base_name}_debug_markers.png"), debug_img)

    # Lưu thông tin chi tiết
    debug_file_path = os.path.join(debug_dir, f"{base_name}_debug_info.txt")
    try:
        with open(debug_file_path, 'w', encoding='utf-8') as f:
            f.write("MARKER DETECTION DEBUG INFO\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total contours analyzed: {len(debug_info)}\n")
            f.write(f"Valid markers found: {len([info for info in debug_info if info['is_valid']])}\n\n")

            for info in debug_info:
                f.write(f"Marker {info['id']}: {'✓ VALID' if info['is_valid'] else '✗ INVALID'}\n")
                f.write(f"  Center: {info['center']}\n")
                f.write(f"  Area: {info['area']:.1f} {'✓' if info['good_area'] else '✗'}\n")
                f.write(f"  Vertices: {info['vertices']} {'✓' if info['vertices'] == 4 else '✗'}\n")
                f.write(f"  Convex: {info['convex']} {'✓' if info['convex'] else '✗'}\n")
                f.write(f"  Aspect ratio: {info['aspect_ratio']:.3f} {'✓' if 0.85 < info['aspect_ratio'] < 1.1 else '✗'}\n")
                f.write(f"  Fill ratio: {info['fill_ratio']:.3f} {'✓' if info['fill_ratio'] > 0.7 else '✗'}\n")
                f.write(f"  Compactness: {info['compactness']:.3f} {'✓' if 0.4 < info['compactness'] < 0.85 else '✗'}\n")
                f.write(f"  Solidity: {info['solidity']:.3f} {'✓' if info['solidity'] > 0.85 else '✗'}\n")
                f.write("\n")
        print(f"Debug info written to: {debug_file_path}")
    except Exception as e:
        print(f"Error writing debug info: {e}")

    print(f"Debug info saved to {debug_dir}/")
    valid_markers = [info for info in debug_info if info['is_valid']]
    print(f"Found {len(valid_markers)} valid markers out of {len(debug_info)} candidates")

    return debug_info

def save_error_images(image, thresh, square_markers, image_path, error_type, ordered_points=None):
    """Lưu ảnh debug khi gặp lỗi"""
    error_dir = "errorLog"
    os.makedirs(error_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(image_path))[0]
    timestamp = str(int(time.time()))

    # Lưu ảnh threshold
    thresh_path = os.path.join(error_dir, f"{base}_{error_type}_{timestamp}_thresh.png")
    Image.fromarray(thresh).save(thresh_path)

    # Tạo ảnh debug với marker được đánh dấu
    debug_image = image.copy()

    # Đánh dấu tất cả marker tìm được
    for i, (cx, cy) in enumerate(square_markers):
        cv2.circle(debug_image, (cx, cy), 15, (0, 255, 0), 3)  # Màu xanh lá
        cv2.putText(debug_image, str(i+1), (cx-10, cy-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Nếu có ordered_points, đánh dấu 4 extreme points
    if ordered_points is not None:
        colors = [(255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]  # Đỏ, Xanh dương, Vàng, Tím
        labels = ["TL", "TR", "BL", "BR"]
        for i, (pt, color, label) in enumerate(zip(ordered_points, colors, labels)):
            cv2.circle(debug_image, tuple(pt.astype(int)), 20, color, -1)
            cv2.putText(debug_image, label, (int(pt[0])-15, int(pt[1])+5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Lưu ảnh debug
    debug_path = os.path.join(error_dir, f"{base}_{error_type}_{timestamp}_debug.png")
    Image.fromarray(cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB)).save(debug_path)

    print(f"🚨 Error images saved: {thresh_path}, {debug_path}")

def count_valid_square_markers(contours):
    """
    Đếm số lượng marker vuông hợp lệ từ danh sách contours

    Args:
        contours: Danh sách contours từ cv2.findContours

    Returns:
        int: Số lượng marker vuông hợp lệ
    """
    valid_markers = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 250 or area > 4000:
            continue

        # Sử dụng epsilon nhỏ hơn để chính xác hơn
        epsilon = 0.015 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx) == 4 and cv2.isContourConvex(approx):
            _, _, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h

            # Kiểm tra thêm độ vuông thực tế
            bbox_area = w * h
            contour_area = cv2.contourArea(cnt)
            fill_ratio = contour_area / bbox_area if bbox_area > 0 else 0

            # Kiểm tra độ compact (gần hình vuông)
            perimeter = cv2.arcLength(cnt, True)
            compactness = 4 * np.pi * contour_area / (perimeter * perimeter) if perimeter > 0 else 0

            # Kiểm tra solidity (độ đặc)
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = contour_area / hull_area if hull_area > 0 else 0

            # Điều kiện strict hơn cho marker vuông - loại bỏ hình tròn
            if (0.7 < aspect_ratio < 1.4 and
                fill_ratio > 0.7 and        # Phải fill ít nhất 70% bounding box
                0.4 < compactness < 0.9 and # Không quá tròn (loại bỏ hình tròn có compactness ~1.0)
                solidity > 0.7):           # Đủ đặc (không có lỗ hổng)
                valid_markers += 1

    return valid_markers

def find_valid_rectangle(contours):
    """
    Từ contours, tìm 4 marker ở góc tạo thành hình chữ nhật hợp lệ.

    Args:
        contours: Danh sách contours từ cv2.findContours.

    Returns:
        tuple: (ordered_points, diagonal_length, count) nếu thành công, (None, 0, 0) nếu thất bại.
    """
    square_markers = []
    # 1. Filter for square-like markers
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 250 or area > 4000: # Mở rộng ngưỡng diện tích để bắt được nhiều marker tiềm năng hơn
            continue

        epsilon = 0.015 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx) == 4 and cv2.isContourConvex(approx):
            _, _, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            bbox_area = w * h
            contour_area = cv2.contourArea(cnt)
            fill_ratio = contour_area / bbox_area if bbox_area > 0 else 0
            perimeter = cv2.arcLength(cnt, True)
            compactness = 4 * np.pi * contour_area / (perimeter * perimeter) if perimeter > 0 else 0
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = contour_area / hull_area if hull_area > 0 else 0

            if (0.7 < aspect_ratio < 1.4 and # Mở rộng ngưỡng tỷ lệ khung hình
                fill_ratio > 0.65 and       # Giảm ngưỡng fill ratio một chút
                0.4 < compactness < 0.9 and # Giữ nguyên
                solidity > 0.75):          # Giảm ngưỡng solidity một chút
                M = cv2.moments(approx)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    corners = approx.reshape(-1, 2)
                    corners_sorted = np.zeros((4, 2), dtype=np.int32)
                    corners_sorted[0] = min(corners, key=lambda p: p[0] + p[1])
                    corners_sorted[1] = max(corners, key=lambda p: p[0] - p[1])
                    corners_sorted[2] = max(corners, key=lambda p: p[1] - p[0])
                    corners_sorted[3] = max(corners, key=lambda p: p[0] + p[1])
                    square_markers.append((cx, cy, corners_sorted))

    marker_count = len(square_markers)
    if marker_count < 4:
        return None, 0, marker_count

    try:
        centers = np.array([(cx, cy) for cx, cy, _ in square_markers])
        tl_idx, tr_idx, bl_idx, br_idx = select_four_extreme_markers_safe(square_markers)
        validate_marker_selection(centers, tl_idx, tr_idx, bl_idx, br_idx, min_distance=50)
    except ValueError:
        return None, 0, marker_count

    tl_marker_corners = square_markers[tl_idx][2]
    tr_marker_corners = square_markers[tr_idx][2]
    bl_marker_corners = square_markers[bl_idx][2]
    br_marker_corners = square_markers[br_idx][2]

    top_left_point = tl_marker_corners[3]
    top_right_point = tr_marker_corners[2]
    bottom_left_point = bl_marker_corners[1]
    bottom_right_point = br_marker_corners[0]

    ordered_points = np.array([top_left_point, top_right_point, bottom_left_point, bottom_right_point], dtype="float32")

    diagonal1 = calculate_distance(top_left_point, bottom_right_point)
    diagonal2 = calculate_distance(top_right_point, bottom_left_point)
    relative_error = abs(diagonal1 - diagonal2) / max(diagonal1, diagonal2)

    if relative_error > 0.05:
        return None, 0, marker_count # Not a valid rectangle

    return ordered_points, max(diagonal1, diagonal2), marker_count

def find_optimal_threshold_and_markers(gray_image):
    """
    Tìm threshold tối ưu bằng cách tìm hình chữ nhật hợp lệ có đường chéo lớn nhất.
    """
    print("🔍 Searching for optimal threshold by finding the largest valid rectangle...")

    best_threshold = -1
    best_diagonal = 0
    best_points = None
    best_thresh_image = None
    results = []

    for threshold_val in range(20, 201, 5):
        _, thresh = cv2.threshold(gray_image, threshold_val, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        ordered_points, diagonal_length, marker_count = find_valid_rectangle(contours)

        results.append({'thresh': threshold_val, 'markers': marker_count, 'diagonal': diagonal_length})

        if ordered_points is not None and diagonal_length > best_diagonal:
            best_diagonal = diagonal_length
            best_threshold = threshold_val
            best_points = ordered_points
            best_thresh_image = thresh.copy()
            print(f"🎯 New best: thresh={threshold_val}, markers={marker_count}, diagonal={diagonal_length:.2f}")

    print("\n📊 Threshold search results:")
    for res in results:
        status = "❌"
        if res['diagonal'] > 0:
            if res['thresh'] == best_threshold:
                status = "🎯 BEST"
            else:
                status = "✅ GOOD"
        print(f"   Threshold {res['thresh']}: {res['markers']} markers, diagonal={res['diagonal']:.1f} {status}")

    if best_points is None:
        # Fallback: if no valid rectangle is found, use the one with most markers as a last resort
        print("\n⚠️ No valid rectangle found. Falling back to threshold with most markers.")
        max_markers = -1
        fallback_thresh = -1
        for res in results:
            if res['markers'] > max_markers:
                max_markers = res['markers']
                fallback_thresh = res['thresh']

        if fallback_thresh != -1:
             print(f"🎯 Fallback selected: threshold {fallback_thresh} with {max_markers} markers.")
             _, best_thresh_image = cv2.threshold(gray_image, fallback_thresh, 255, cv2.THRESH_BINARY_INV)
             return fallback_thresh, best_thresh_image, None # No points, force re-detection
        else:
            return -1, None, None # Complete failure

    print(f"\n🎯 Optimal threshold found: {best_threshold} with diagonal length {best_diagonal:.2f}")
    return best_threshold, best_thresh_image, best_points

def process_image(image_path, output_dir, save_debug=False):
    """
    Xử lý ảnh trực tiếp không qua shadow removal

    Args:
        image_path: Đường dẫn ảnh đầu vào
        output_dir: Thư mục đầu ra
        save_debug: Có lưu ảnh debug không
    """
    print(f"🔄 Processing: {image_path}")

    # Đọc ảnh gốc
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Không thể đọc ảnh: {image_path}")

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    print(f"📏 Image size: {image.shape}")
    print(f"🚀 Processing directly without shadow removal")

    # Chuyển sang grayscale để detect marker
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Tìm threshold tối ưu và các điểm marker tương ứng
    optimal_thresh, thresh, ordered_points = find_optimal_threshold_and_markers(gray)

    if optimal_thresh == -1 or thresh is None:
        # Lưu ảnh lỗi nếu không tìm thấy threshold phù hợp
        save_error_images(image, gray, [], image_path, "no_threshold_found")
        raise ValueError("Không thể tìm thấy bất kỳ threshold phù hợp nào để phát hiện marker.")

    # Nếu không tìm thấy hình chữ nhật hợp lệ, fallback để thử lại với threshold tốt nhất
    if ordered_points is None:
        print("Retrying marker detection with fallback threshold...")
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ordered_points, _, _ = find_valid_rectangle(contours)

    if ordered_points is None:
        # Vẫn không thành công ngay cả với fallback
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        square_markers = []
        for cnt in contours:
             if 1000 < cv2.contourArea(cnt) < 3200:
                 M = cv2.moments(cnt)
                 if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    square_markers.append((cx,cy))
        save_error_images(image, thresh, square_markers, image_path, "insufficient_markers")
        raise ValueError("Không tìm đủ 4 marker hợp lệ để thực hiện perspective correction.")

    # Debug marker detection nếu cần
    if save_debug:
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        debug_marker_detection(image, thresh, contours, image_path)

    width, height = 800, 1131
    # Điều chỉnh destination points để cắt thêm vào bên trong
    margin_left = 10   # Cắt thêm 10px vào bên trong ở phía trái
    margin_top = 7    # Cắt thêm 15px vào bên trong ở phía trên
    margin_right = 4   # Cắt thêm 3px vào bên trong ở phía phải

    dst_pts = np.array([
        [-margin_left, -margin_top],                    # top-left: dịch ra ngoài để cắt vào trong
        [width - 1 + margin_right, -margin_top],        # top-right: dịch lên trên và sang phải
        [-margin_left, height - 1],                     # bottom-left: chỉ dịch sang trái
        [width - 1 + margin_right, height - 1]          # bottom-right: dịch sang phải
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(ordered_points, dst_pts)
    # Sử dụng INTER_CUBIC cho chất lượng tốt hơn, tránh mờ ảnh
    warped = cv2.warpPerspective(image, M, (width, height), flags=cv2.INTER_CUBIC)

    # Debug ảnh marker
    debug_image = image.copy()
    for pt in ordered_points:
        cv2.circle(debug_image, tuple(pt.astype(int)), 10, (0, 0, 255), -1)

    os.makedirs(output_dir, exist_ok=True)

    # Tạo tên file đầu ra
    output_flattened = os.path.join(output_dir, f"{base_name}_flattened.png")
    output_debug = os.path.join(output_dir, f"{base_name}_debug.png")
    output_thresh = os.path.join(output_dir, f"{base_name}_thresh.png")

    # Lưu kết quả
    Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)).save(output_flattened)
    Image.fromarray(cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB)).save(output_debug)
    Image.fromarray(thresh).save(output_thresh)

    print(f"✅ Main outputs saved:")
    print(f"   Flattened: {output_flattened}")
    print(f"   Debug: {output_debug}")
    print(f"   Threshold: {output_thresh}")

    return output_flattened

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Cắt và làm phẳng ảnh trắc nghiệm từ 4 marker đen",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Xử lý ảnh trực tiếp (tự động tìm threshold tối ưu)
  python omr_marker_flatten_robust.py image.jpg

  # Lưu ảnh debug
  python omr_marker_flatten_robust.py image.jpg --save-debug
        """
    )

    parser.add_argument("image_path", help="Đường dẫn ảnh đầu vào")
    parser.add_argument("--output_dir", default="output", help="Thư mục chứa ảnh đầu ra")
    parser.add_argument("--save-debug", action='store_true',
                       help="Lưu ảnh debug")

    args = parser.parse_args()

    print("🎯 OMR Marker Flatten Direct Processing v4.0")
    print("=" * 50)

    try:
        result_path = process_image(
            args.image_path,
            args.output_dir,
            save_debug=args.save_debug
        )
        print(f"\n🎉 Processing completed successfully!")
        print(f"📄 Main result: {result_path}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("💡 Try using --save-debug to see debug images")
        exit(1)
