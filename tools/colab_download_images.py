import hashlib
import os
import shutil
import subprocess
import zipfile

URL = "https://ndownloader.figshare.com/files/50010117"
ARCHIVE = "/content/gastrohun-data/Labeled Images.zip"
EXPECTED_SIZE = 2_820_524_449
EXPECTED_MD5 = "0d6b48a36a21388123cc5d0231dbd362"
IMAGE_DIR = "/content/gastrohun-data/images"

os.makedirs(IMAGE_DIR, exist_ok=True)

if not os.path.exists(ARCHIVE) or os.path.getsize(ARCHIVE) != EXPECTED_SIZE:
    subprocess.run(
        ["wget", "--continue", "--show-progress", "--output-document", ARCHIVE, URL],
        check=True,
    )

digest = hashlib.md5()
with open(ARCHIVE, "rb") as stream:
    for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
        digest.update(block)
if digest.hexdigest() != EXPECTED_MD5:
    raise RuntimeError(f"MD5 mismatch: {digest.hexdigest()} != {EXPECTED_MD5}")

extracted = 0
with zipfile.ZipFile(ARCHIVE) as archive:
    for member in archive.infolist():
        if member.is_dir() or not member.filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        destination = os.path.join(IMAGE_DIR, os.path.basename(member.filename))
        with archive.open(member) as source, open(destination, "wb") as target:
            shutil.copyfileobj(source, target, length=16 * 1024 * 1024)
        extracted += 1

print(f"archive_verified_bytes: {os.path.getsize(ARCHIVE)}")
print(f"images_extracted: {extracted}")
