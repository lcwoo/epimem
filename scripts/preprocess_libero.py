from PIL import Image
import os
import cv2
import numpy as np
import tensorflow_datasets as tfds
from tqdm import tqdm

# ===== 설정 =====
INPUT_DATASET_DIR = "datasets/modified_libero_rlds/libero_10_no_noops/1.0.0"
OUTPUT_DIR = "datasets/preview_png"
FINAL_SIZE = 512
TILE_SIZE = 128
NUM_EPISODES = 1

# 타일 위치 정의
image_tile_positions = [
    (0, 0),  # Img1
    (1, 0),  # Img2
    (2, 0),  # Img3
    (3, 0),  # Img4
    (0, 1),  # Img5
    (0, 2),  # Img6
    (0, 3),  # Img7
]

# NEWEST 영역: (1,1) ~ (3,3)
newest_area_origin = (1 * TILE_SIZE, 1 * TILE_SIZE)
newest_area_size = (3 * TILE_SIZE, 3 * TILE_SIZE)

# ===== 데이터셋 로드 =====
builder = tfds.builder_from_directory(INPUT_DATASET_DIR)
builder.download_and_prepare()
dataset = builder.as_dataset(split="train")
os.makedirs(OUTPUT_DIR, exist_ok=True)

for idx, episode in tqdm(enumerate(dataset), desc="Processing episodes"):
    if idx >= NUM_EPISODES:
        break

    episode_dir = os.path.join(OUTPUT_DIR, f"episode_{idx:04d}")
    os.makedirs(episode_dir, exist_ok=True)

    # 이미지 수집
    images = []
    for step in episode["steps"]:
        obs = step["observation"]
        img = obs["image"].numpy()
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        rgb_blurred = cv2.cvtColor(blurred, cv2.COLOR_GRAY2RGB)
        images.append(rgb_blurred)

    if len(images) < 8:
        continue

    selected_images = images[-8:]  # 7장 + 최신 이미지
    previous_imgs = selected_images[:-1]
    newest_img = selected_images[-1]

    canvas = Image.new("RGB", (FINAL_SIZE, FINAL_SIZE), (255, 255, 255))

    # 이전 이미지 배치
    for img_np, (x_idx, y_idx) in zip(previous_imgs, image_tile_positions):
        x, y = x_idx * TILE_SIZE, y_idx * TILE_SIZE
        patch = Image.fromarray(img_np).resize((TILE_SIZE, TILE_SIZE))
        canvas.paste(patch, (x, y))

    # 최신 이미지 배치: 3x3 영역
    newest_patch = Image.fromarray(newest_img).resize(newest_area_size)
    canvas.paste(newest_patch, newest_area_origin)

    save_path = os.path.join(episode_dir, "highlighted_memory_grid.png")
    canvas.save(save_path)
    print(f"[✓] Saved → {save_path}")

print(f"\n✅ 완료: {OUTPUT_DIR}/episode_*/highlighted_memory_grid.png")
