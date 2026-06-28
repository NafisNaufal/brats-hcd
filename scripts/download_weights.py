"""Download pretrained weights for Swin and ResNet50 encoders."""

import os

cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "brats_weights")
os.makedirs(cache_dir, exist_ok=True)


def download_swin():
    import urllib.request
    path = os.path.join(cache_dir, "model_swinvit.pt")
    if os.path.isfile(path):
        print(f"[Swin] Already exists: {path}")
        return
    url = "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/model_swinvit.pt"
    print("[Swin] Downloading...")
    urllib.request.urlretrieve(url, path)
    print(f"[Swin] Saved to {path}")


def download_med3d():
    import urllib.request
    path = os.path.join(cache_dir, "resnet_50_med3d.pth")
    if os.path.isfile(path):
        print(f"[Med3D] Already exists: {path}")
        return
    url = "https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet50/resolve/main/resnet_50.pth"
    print("[Med3D] Downloading ResNet50 weights from HuggingFace...")
    urllib.request.urlretrieve(url, path)
    print(f"[Med3D] Saved to {path}")


if __name__ == "__main__":
    download_swin()
    download_med3d()
    print("Done.")
