import os 
import shutil 
import kagglehub

DATASET_NAME = "mlg-ulb/creditcardfraud"
LOCAL_DATASET_DIR = "/src/data/dataset"


def download():
    """Загружает датасет с использованием kagglehub и копирует архив в локальную директорию датасета."""
    print("Загрузка датасета из Kaggle...")
    path = kagglehub.dataset_download(DATASET_NAME)
    print(f"Путь к файлам датасета: {path}")

    os.makedirs(LOCAL_DATASET_DIR, exist_ok=True)
    for filename in os.listdir(path):
        full_src_path = os.path.join(path, filename)
        full_dst_path = os.path.join(LOCAL_DATASET_DIR, filename)
        shutil.copy(full_src_path, full_dst_path)
        print(f"Скопирован {filename} в {full_dst_path}")


if __name__=="__main__":
    download()