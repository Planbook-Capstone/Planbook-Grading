# Test parsing đáp án học sinh

from api.answer_marking_service import parse_student_answers

def test_parse_student_answers():
    """Test hàm parse_student_answers"""
    
    # Dữ liệu test giả lập student_answers từ detect_circles
    sample_student_answers = [
        # Part 1
        "part1_1_a_236_702",  # Câu 1, đáp án A
        "part1_2_b_280_702",  # Câu 2, đáp án B
        "part1_3_c_324_702",  # Câu 3, đáp án C
        
        # Part 2  
        "part2_1_a_D_238_1054",  # Câu 1a, đáp án D
        "part2_1_b_D_282_1054",  # Câu 1b, đáp án D
        "part2_1_c_S_326_1054",  # Câu 1c, đáp án S
        "part2_1_d_S_370_1054",  # Câu 1d, đáp án S
        "part2_2_a_D_238_1090",  # Câu 2a, đáp án D
        "part2_2_b_S_282_1090",  # Câu 2b, đáp án S
        
        # Part 3
        "part3_1_minus_1_354_1270",  # Câu 1, dấu trừ ở vị trí 1
        "part3_1_3_2_378_1270",      # Câu 1, số 3 ở vị trí 2
        "part3_1_comma_3_402_1270",  # Câu 1, dấu phẩy ở vị trí 3
        "part3_1_2_4_426_1270",      # Câu 1, số 2 ở vị trí 4
        
        "part3_2_1_1_354_1294",      # Câu 2, số 1 ở vị trí 1
        "part3_2_2_2_378_1294",      # Câu 2, số 2 ở vị trí 2
        "part3_2_3_3_402_1294",      # Câu 2, số 3 ở vị trí 3
        
        "part3_3_2_1_354_1318",      # Câu 3, số 2 ở vị trí 1
        "part3_3_0_2_378_1318",      # Câu 3, số 0 ở vị trí 2
    ]
    
    print("🧪 Testing parse_student_answers function")
    print("=" * 50)
    
    result = parse_student_answers(sample_student_answers)
    
    print("📋 Parsed student answers:")
    print(f"Part 1: {result['part1']}")
    print(f"Part 2: {result['part2']}")
    print(f"Part 3: {result['part3']}")
    
    print("\n✅ Expected results:")
    expected = {
        "part1": {
            "1": "A",
            "2": "B",
            "3": "C"
        },
        "part2": {
            "1a": "D",
            "1b": "D",
            "1c": "S", 
            "1d": "S",
            "2a": "D",
            "2b": "S"
        },
        "part3": {
            "1": "-3,2",  # minus(1) + 3(2) + comma(3) + 2(4)
            "2": "123",   # 1(1) + 2(2) + 3(3)
            "3": "20"     # 2(1) + 0(2)
        }
    }
    
    print(f"Part 1 expected: {expected['part1']}")
    print(f"Part 2 expected: {expected['part2']}")
    print(f"Part 3 expected: {expected['part3']}")
    
    # Kiểm tra kết quả
    print("\n🔍 Verification:")
    print(f"Part 1 match: {result['part1'] == expected['part1']}")
    print(f"Part 2 match: {result['part2'] == expected['part2']}")
    print(f"Part 3 match: {result['part3'] == expected['part3']}")

if __name__ == "__main__":
    test_parse_student_answers()
