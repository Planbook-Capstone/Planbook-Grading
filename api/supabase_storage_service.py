"""
Supabase Storage Service
Service để upload ảnh lên Supabase Storage
"""

import os
import uuid
import logging
from typing import Optional, Tuple, Dict, Any
from datetime import datetime
from supabase import create_client, Client
import mimetypes

# Configure logging
logger = logging.getLogger(__name__)

class SupabaseStorageService:
    """Service để quản lý upload ảnh lên Supabase Storage"""
    
    def __init__(self, url: str, key: str, bucket_name: str):
        """
        Khởi tạo Supabase Storage Service
        
        Args:
            url: Supabase project URL
            key: Supabase API key (anon key)
            bucket_name: Tên bucket để lưu trữ ảnh
        """
        self.url = url
        self.key = key
        self.bucket_name = bucket_name
        self.supabase: Client = create_client(url, key)
        
        logger.info(f"Initialized Supabase Storage Service with bucket: {bucket_name}")
    
    def upload_image(self, file_path: str, folder: str = "images") -> Dict[str, Any]:
        """
        Upload ảnh lên Supabase Storage
        
        Args:
            file_path: Đường dẫn đến file ảnh cần upload
            folder: Thư mục trong bucket để lưu ảnh (mặc định: "images")
            
        Returns:
            Dict chứa thông tin về file đã upload:
            {
                "success": bool,
                "file_url": str,
                "file_path": str,
                "file_name": str,
                "message": str,
                "error": str (nếu có lỗi)
            }
        """
        try:
            # Kiểm tra file có tồn tại không
            if not os.path.exists(file_path):
                return {
                    "success": False,
                    "error": f"File không tồn tại: {file_path}",
                    "message": "File không tồn tại"
                }
            
            # Lấy thông tin file
            file_name = os.path.basename(file_path)
            file_extension = os.path.splitext(file_name)[1].lower()
            
            # Tạo tên file unique với timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            unique_file_name = f"{timestamp}_{unique_id}{file_extension}"
            
            # Đường dẫn trong bucket
            storage_path = f"{folder}/{unique_file_name}"
            
            # Đọc file
            with open(file_path, 'rb') as f:
                file_data = f.read()
            
            # Xác định content type
            content_type, _ = mimetypes.guess_type(file_path)
            if not content_type:
                content_type = 'image/jpeg'  # Default
            
            logger.info(f"Uploading file: {file_name} -> {storage_path}")
            logger.info(f"File size: {len(file_data)} bytes, Content-Type: {content_type}")

            # Upload file lên Supabase Storage
            try:
                # Thử upload với file_options
                response = self.supabase.storage.from_(self.bucket_name).upload(
                    path=storage_path,
                    file=file_data,
                    file_options={"content-type": content_type}
                )
            except Exception as e:
                # Nếu lỗi với file_options, thử upload đơn giản
                logger.warning(f"Upload with file_options failed, trying simple upload: {e}")
                response = self.supabase.storage.from_(self.bucket_name).upload(
                    path=storage_path,
                    file=file_data
                )
            
            # Lấy public URL
            public_url = self.supabase.storage.from_(self.bucket_name).get_public_url(storage_path)
            
            logger.info(f"Upload successful: {public_url}")
            
            return {
                "success": True,
                "file_url": public_url,
                "file_path": storage_path,
                "file_name": unique_file_name,
                "original_name": file_name,
                "content_type": content_type,
                "file_size": len(file_data),
                "message": "Upload ảnh thành công"
            }
            
        except Exception as e:
            error_msg = f"Lỗi upload ảnh: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "message": "Upload ảnh thất bại"
            }
    
    def upload_multiple_images(self, file_paths: list, folder: str = "images") -> Dict[str, Any]:
        """
        Upload nhiều ảnh cùng lúc
        
        Args:
            file_paths: List đường dẫn các file ảnh
            folder: Thư mục trong bucket để lưu ảnh
            
        Returns:
            Dict chứa thông tin về các file đã upload
        """
        results = []
        success_count = 0
        error_count = 0
        
        for file_path in file_paths:
            result = self.upload_image(file_path, folder)
            results.append(result)
            
            if result["success"]:
                success_count += 1
            else:
                error_count += 1
        
        return {
            "success": error_count == 0,
            "total_files": len(file_paths),
            "success_count": success_count,
            "error_count": error_count,
            "results": results,
            "message": f"Upload hoàn tất: {success_count}/{len(file_paths)} file thành công"
        }
    
    def delete_image(self, file_path: str) -> Dict[str, Any]:
        """
        Xóa ảnh từ Supabase Storage
        
        Args:
            file_path: Đường dẫn file trong bucket (ví dụ: "images/filename.jpg")
            
        Returns:
            Dict chứa thông tin về việc xóa file
        """
        try:
            response = self.supabase.storage.from_(self.bucket_name).remove([file_path])
            
            logger.info(f"Deleted file: {file_path}")
            
            return {
                "success": True,
                "file_path": file_path,
                "message": "Xóa ảnh thành công"
            }
            
        except Exception as e:
            error_msg = f"Lỗi xóa ảnh: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "message": "Xóa ảnh thất bại"
            }
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """
        Lấy thông tin file từ Supabase Storage
        
        Args:
            file_path: Đường dẫn file trong bucket
            
        Returns:
            Dict chứa thông tin file
        """
        try:
            # Lấy public URL
            public_url = self.supabase.storage.from_(self.bucket_name).get_public_url(file_path)
            
            return {
                "success": True,
                "file_path": file_path,
                "file_url": public_url,
                "message": "Lấy thông tin file thành công"
            }
            
        except Exception as e:
            error_msg = f"Lỗi lấy thông tin file: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "message": "Lấy thông tin file thất bại"
            }
    
    def list_files(self, folder: str = "", limit: int = 100) -> Dict[str, Any]:
        """
        Liệt kê các file trong bucket

        Args:
            folder: Thư mục cần liệt kê (để trống để liệt kê tất cả)
            limit: Số lượng file tối đa trả về

        Returns:
            Dict chứa danh sách file
        """
        try:
            # Sử dụng list() đơn giản hơn
            response = self.supabase.storage.from_(self.bucket_name).list(folder)
            
            files = []
            for file_info in response:
                files.append({
                    "name": file_info.get("name"),
                    "id": file_info.get("id"),
                    "updated_at": file_info.get("updated_at"),
                    "created_at": file_info.get("created_at"),
                    "last_accessed_at": file_info.get("last_accessed_at"),
                    "metadata": file_info.get("metadata", {})
                })
            
            return {
                "success": True,
                "files": files,
                "count": len(files),
                "message": f"Tìm thấy {len(files)} file"
            }
            
        except Exception as e:
            error_msg = f"Lỗi liệt kê file: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "message": "Liệt kê file thất bại"
            }


# Tạo instance global với config mặc định
SUPABASE_URL = "https://zzfjygzmhvvsvycmvrlm.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inp6Zmp5Z3ptaHZ2c3Z5Y212cmxtIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDIwMzIwNzcsImV4cCI6MjA1NzYwODA3N30.1gslu7VM7EYV7lsr5QlKjdPnpvswF6dx8S5lb00nwcU"
SUPABASE_BUCKET = "biteologystorage"

# Khởi tạo service
supabase_storage = SupabaseStorageService(
    url=SUPABASE_URL,
    key=SUPABASE_KEY,
    bucket_name=SUPABASE_BUCKET
)


def upload_image_to_supabase(file_path: str, folder: str = "images") -> Dict[str, Any]:
    """
    Hàm tiện ích để upload ảnh lên Supabase
    
    Args:
        file_path: Đường dẫn file ảnh
        folder: Thư mục trong bucket
        
    Returns:
        Dict chứa thông tin upload
    """
    return supabase_storage.upload_image(file_path, folder)


def upload_multiple_images_to_supabase(file_paths: list, folder: str = "images") -> Dict[str, Any]:
    """
    Hàm tiện ích để upload nhiều ảnh lên Supabase
    
    Args:
        file_paths: List đường dẫn các file ảnh
        folder: Thư mục trong bucket
        
    Returns:
        Dict chứa thông tin upload
    """
    return supabase_storage.upload_multiple_images(file_paths, folder)
