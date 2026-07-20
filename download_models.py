"""
download_models.py
Run this ONCE to download the age/gender model files needed by detector.py.
No TensorFlow needed — these are lightweight Caffe models loaded via OpenCV's
built-in DNN module.

Usage:
    python download_models.py
"""

import os
import urllib.request

BASE_URL = "https://raw.githubusercontent.com/smahesh29/Gender-and-Age-Detection/master/"

FILES = [
    "age_deploy.prototxt",
    "age_net.caffemodel",
    "gender_deploy.prototxt",
    "gender_net.caffemodel",
]


def download_file(filename):
    if os.path.exists(filename):
        print(f"[SKIP] {filename} already exists.")
        return
    url = BASE_URL + filename
    print(f"[DOWNLOADING] {filename} ...")
    try:
        urllib.request.urlretrieve(url, filename)
        print(f"[DONE] {filename}")
    except Exception as e:
        print(f"[ERROR] Failed to download {filename}: {e}")
        print(f"        Try downloading it manually from: {url}")


if __name__ == "__main__":
    for f in FILES:
        download_file(f)
    print("\nAll done. You can now run: python detector.py --source 0")
