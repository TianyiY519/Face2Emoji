import kagglehub
import shutil
import os

def download_fer2013():
    print("Downloading FER2013 from Kaggle...")
    path = kagglehub.dataset_download("msambare/fer2013")
    print("Downloaded to:", path)

    # Expected folders: train, test
    train_dir = os.path.join(path, "train")
    test_dir = os.path.join(path, "test")

    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        raise FileNotFoundError("train/test folders not found in downloaded dataset. This is the image version of FER2013.")

    # Prepare destination folder
    dest_root = "data/fer2013"
    os.makedirs(dest_root, exist_ok=True)

    # Copy train folder
    dest_train = os.path.join(dest_root, "train")
    if os.path.exists(dest_train):
        shutil.rmtree(dest_train)
    shutil.copytree(train_dir, dest_train)

    # Copy test folder
    dest_test = os.path.join(dest_root, "test")
    if os.path.exists(dest_test):
        shutil.rmtree(dest_test)
    shutil.copytree(test_dir, dest_test)

    print("Copied train/test to:", dest_root)

if __name__ == "__main__":
    download_fer2013()

