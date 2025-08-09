import kaggle
import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

def download_and_unzip_kaggle_dataset(dataset_api_name, download_path='.'):
    """
    Tải và giải nén một dataset từ Kaggle.

    Args:
        dataset_api_name (str): Tên API của dataset (ví dụ: 'user/dataset-name').
        download_path (str): Thư mục để lưu trữ dữ liệu.
    """
    # Tạo thư mục nếu chưa tồn tại
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    print(f"Bắt đầu tải dataset: {dataset_api_name}...")

    # Khởi tạo API
    api = KaggleApi()
    api.authenticate()

    # Tải dataset (sẽ là một file .zip)
    api.dataset_download_files(dataset_api_name, path=download_path, quiet=False)
    print("Tải file zip thành công.")

    # Tìm file zip vừa tải
    zip_file_path = os.path.join(download_path, f"{dataset_api_name.split('/')[1]}.zip")

    # Giải nén file zip
    if os.path.exists(zip_file_path):
        print(f"Bắt đầu giải nén {zip_file_path}...")
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(download_path)
        print("Giải nén thành công.")

        # Xóa file zip sau khi giải nén (tùy chọn)
        os.remove(zip_file_path)
        print("Đã xóa file zip.")
    else:
        print(f"Lỗi: Không tìm thấy file zip tại {zip_file_path}")

if __name__ == "__main__":
    # Tên API của dataset bạn muốn tải
    KAGGLE_DATASET_NAME = "chethuhn/network-intrusion-dataset"
    # Thư mục để chứa dữ liệu, ví dụ 'data'
    DATA_DIRECTORY = "cicids2017_data"

    download_and_unzip_kaggle_dataset(KAGGLE_DATASET_NAME, DATA_DIRECTORY)
