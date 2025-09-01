from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
import uvicorn
import os
import uuid
import logging
import requests
import tempfile
import json
from urllib.parse import urlparse
from api.main_service import process_image as process_main_image
from api.omr_service import process_image as process_omr_image
from api.circle_detection_service import detect_circles
from api.answer_marking_service import mark_correct_answers_on_image, create_answer_summary

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app with increased file size limits
app = FastAPI(
    title="OMR API",
    description="API for processing OMR (Optical Mark Recognition) images",
)

# Add CORS middleware to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Pydantic models for request body
class ImageUrlRequest(BaseModel):
    image_url: str

class CorrectAnswersRequest(BaseModel):
    correct_answers: dict

UPLOAD_DIR = "temp_uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

@app.get("/")
def read_root():
    return {"message": "Welcome to the OMR API"}


@app.post("/process-image/")
async def create_upload_file(file: UploadFile = File(..., description="Image file to process (no size limit)")):
    file_path = None
    flattened_image_path = None
    try:
        # Create a unique filename
        file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)

        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        # Process the image with omr_service
        flattened_image_path = process_omr_image(file_path, UPLOAD_DIR)

        # Process the flattened image with main_service
        results = process_main_image(flattened_image_path)

        return JSONResponse(content=results)

    except Exception as e:
        # Log the actual error for debugging purposes
        logger.error(f"Error processing image {file.filename}: {str(e)}")

        # Return user-friendly Vietnamese error message
        error_message = "Vui lòng chụp lại ảnh rõ nét, đủ ánh sáng và đủ các marker!"
        return JSONResponse(
            status_code=400,
            content={"error": error_message, "message": error_message}
        )
    finally:
        # Clean up the uploaded files
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
        if flattened_image_path and os.path.exists(flattened_image_path):
            os.remove(flattened_image_path)


@app.post("/process-image-url/")
async def process_image_from_url(request: ImageUrlRequest):
    """
    API endpoint để xử lý ảnh từ URL thay vì upload file

    Args:
        request: ImageUrlRequest chứa image_url

    Returns:
        JSONResponse: Kết quả xử lý OMR hoặc thông báo lỗi
    """
    temp_file_path = None
    flattened_image_path = None

    try:
        # Validate URL
        if not request.image_url or not request.image_url.strip():
            raise HTTPException(status_code=400, detail="URL ảnh không được để trống")

        # Parse URL để lấy extension
        parsed_url = urlparse(request.image_url)
        path = parsed_url.path

        # Lấy extension từ URL, nếu không có thì mặc định là .jpg
        file_extension = os.path.splitext(path)[1] if os.path.splitext(path)[1] else '.jpg'

        # Tạo tên file tạm thời
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        temp_file_path = os.path.join(UPLOAD_DIR, unique_filename)

        # Tải ảnh từ URL
        logger.info(f"Downloading image from URL: {request.image_url}")

        # Set headers để giả lập browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        # Tải ảnh với timeout
        response = requests.get(request.image_url, headers=headers, timeout=30, stream=True)
        response.raise_for_status()

        # Kiểm tra content type
        content_type = response.headers.get('content-type', '').lower()
        if not any(img_type in content_type for img_type in ['image/jpeg', 'image/jpg', 'image/png', 'image/bmp', 'image/tiff']):
            raise HTTPException(status_code=400, detail="URL không trả về ảnh hợp lệ")

        # Lưu ảnh vào file tạm thời
        with open(temp_file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        logger.info(f"Image downloaded successfully: {temp_file_path}")

        # Kiểm tra kích thước file
        file_size = os.path.getsize(temp_file_path)
        if file_size == 0:
            raise HTTPException(status_code=400, detail="File ảnh tải về có kích thước 0 bytes")

        logger.info(f"Downloaded file size: {file_size} bytes")

        # Xử lý ảnh với omr_service
        flattened_image_path = process_omr_image(temp_file_path, UPLOAD_DIR)

        # Xử lý ảnh đã flatten với main_service
        results = process_main_image(flattened_image_path)

        # Thêm thông tin URL vào kết quả
        results["source_url"] = request.image_url
        results["processing_method"] = "url_download"

        return JSONResponse(content=results)

    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading image from URL {request.image_url}: {str(e)}")
        return JSONResponse(
            status_code=400,
            content={"error": "Không thể tải ảnh từ URL. Vui lòng kiểm tra lại đường dẫn!", "message": f"Lỗi tải ảnh: {str(e)}"}
        )
    except HTTPException as e:
        return JSONResponse(
            status_code=e.status_code,
            content={"error": e.detail, "message": e.detail}
        )
    except Exception as e:
        logger.error(f"Error processing image from URL {request.image_url}: {str(e)}")

        # Return user-friendly Vietnamese error message
        error_message = "Vui lòng chụp lại ảnh rõ nét, đủ ánh sáng và đủ các marker!"
        return JSONResponse(
            status_code=400,
            content={"error": error_message, "message": error_message}
        )
    finally:
        # Clean up temporary files
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.info(f"Cleaned up temporary file: {temp_file_path}")
            except Exception as e:
                logger.warning(f"Failed to remove temporary file {temp_file_path}: {e}")

        if flattened_image_path and os.path.exists(flattened_image_path):
            try:
                os.remove(flattened_image_path)
                logger.info(f"Cleaned up flattened image: {flattened_image_path}")
            except Exception as e:
                logger.warning(f"Failed to remove flattened image {flattened_image_path}: {e}")


@app.post("/detect-circles/")
async def detect_circles_endpoint(file: UploadFile = File(..., description="Image file to detect circles in"), debug: bool = False):
    file_path = None
    try:
        # Create a unique filename
        file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)

        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        # Process the image with circle_detection_service
        detection_results, debug_image_path = detect_circles(file_path, debug=debug)

        response_data = {
            "all_answers": detection_results.get("all_answers", []),
            "student_answers": detection_results.get("student_answers", []),
            "debug_image_path": debug_image_path
        }

        return JSONResponse(content=response_data)

    except Exception as e:
        # Log the actual error for debugging purposes
        logger.error(f"Error processing image {file.filename} for circle detection: {str(e)}")

        # Return user-friendly Vietnamese error message
        error_message = "Có lỗi xảy ra trong quá trình nhận diện hình tròn."
        return JSONResponse(
            status_code=400,
            content={"error": error_message, "message": str(e)}
        )
    finally:
        # Clean up the uploaded file
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
        # The debug image is not removed automatically, as the user might want to access it.
        # A more robust solution would involve a cleanup mechanism for debug files.


@app.post("/mark-correct-answers/")
async def mark_correct_answers_endpoint(
    file: UploadFile = File(..., description="Image file to mark correct answers on"),
    correct_answers: str = Form(..., description="JSON string containing correct answers")
):
    """
    API endpoint để đánh dấu đáp án đúng lên ảnh
    
    Args:
        file: File ảnh upload
        correct_answers: JSON string chứa đáp án đúng
        
    Returns:
        JSONResponse chứa đường dẫn ảnh đã đánh dấu và thông tin summary
    """
    file_path = None
    marked_image_path = None
    
    try:
        # Parse JSON đáp án đúng
        try:
            correct_answers_dict = json.loads(correct_answers)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON format for correct_answers: {str(e)}")
        
        # Nếu JSON có cấu trúc {"correct_answers": {...}} thì lấy phần bên trong
        # Nếu không thì sử dụng trực tiếp
        if "correct_answers" in correct_answers_dict:
            final_answers = correct_answers_dict["correct_answers"]
        else:
            final_answers = correct_answers_dict
        
        # Create a unique filename
        file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)

        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        # Đánh dấu đáp án đúng lên ảnh
        marked_image_path = mark_correct_answers_on_image(
            file_path, 
            final_answers, 
            output_dir="result"
        )
        
        # Tạo summary thông tin
        detection_results, _ = detect_circles(file_path, debug=False)
        all_circles = detection_results.get("all_answers", [])
        student_answers = detection_results.get("student_answers", [])
        summary = create_answer_summary(final_answers, all_circles, student_answers)

        response_data = {
            "marked_image_path": marked_image_path,
            "summary": summary,
            "message": "Đánh dấu đáp án đúng thành công"
        }

        return JSONResponse(content=response_data)

    except HTTPException as e:
        raise e
    except Exception as e:
        # Log the actual error for debugging purposes
        logger.error(f"Error marking correct answers on image {file.filename}: {str(e)}")

        # Return user-friendly Vietnamese error message
        error_message = "Có lỗi xảy ra trong quá trình đánh dấu đáp án đúng."
        return JSONResponse(
            status_code=400,
            content={"error": error_message, "message": str(e)}
        )
    finally:
        # Clean up the uploaded file
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
        # marked_image_path is kept for user to access the result

if __name__ == "__main__":
    # Configure uvicorn with increased limits for large file uploads
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        # Remove file size limits
        limit_max_requests=1000,
        limit_concurrency=1000,
        # Increase timeout for large file processing
        timeout_keep_alive=300,
        # Allow larger request bodies (set to 1GB)
        h11_max_incomplete_event_size=1024*1024*1024
    )

