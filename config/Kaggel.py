import os
import shutil
import kagglehub
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from logger import get_logger  
logger = get_logger()  

def download_and_store_dataset(dataset_name: str, raw_dir: str = "data/raw"):
    try:
        # Download dataset
        downloaded_path = kagglehub.dataset_download(dataset_name)
        logger.info(f"Dataset downloaded to: {downloaded_path}")

        # Check if the downloaded path is valid
        if not downloaded_path or not os.path.exists(downloaded_path):
            raise FileNotFoundError(f"Downloaded path {downloaded_path} does not exist or is invalid.")
        logger.info(f"Valid downloaded path confirmed: {downloaded_path}")

        # Ensure target directory exists
        os.makedirs(raw_dir, exist_ok=True)

        if os.path.isdir(downloaded_path):
            # If it's a directory, move all files inside it up to raw_dir
            try:
                for item in os.listdir(downloaded_path):
                    src = os.path.join(downloaded_path, item)
                    dst = os.path.join(raw_dir, item)
                    shutil.move(src, dst)
                    logger.debug(f"Moved {src} -> {dst}")
                os.rmdir(downloaded_path)  # Remove now empty folder
                logger.info(f"Moved files from {downloaded_path} to {raw_dir} and removed folder")
            except Exception as e:
                logger.error(f"Failed moving files from subfolder: {e}", exc_info=True)
                raise
        else:
            # If it's a single file, move it to raw_dir
            try:
                filename = os.path.basename(downloaded_path)
                dest_path = os.path.join(raw_dir, filename)
                shutil.move(downloaded_path, dest_path)
                logger.info(f"Moved file {filename} to {raw_dir}")
            except Exception as e:
                logger.error(f"Failed moving file: {e}", exc_info=True)
                raise

    except Exception as e:
        logger.error(f"Failed to download or process dataset: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    download_and_store_dataset("zedekiaobuya/housingdatacsv")
