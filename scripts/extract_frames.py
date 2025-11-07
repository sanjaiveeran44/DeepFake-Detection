import cv2
import os
from tqdm import tqdm

# Input and output directories
VIDEO_ROOT = "../videos"         # contains real/ and fake/
OUTPUT_ROOT = "../dataset"       # where frames will be saved (real/ and fake/)
MAX_FRAMES = 90               # upper limit per video
SKIP_FRAMES = 2               # extract every 2nd frame to avoid duplicates

def extract_frames(video_path, save_dir, max_frames=90, skip_frames=2):
    os.makedirs(save_dir, exist_ok=True)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # If already extracted, skip
    video_output_folder = os.path.join(save_dir, video_name)
    if os.path.exists(video_output_folder) and len(os.listdir(video_output_folder)) >= max_frames:
        print(f"✅ Skipping {video_name}, already processed!")
        return
    
    os.makedirs(video_output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or saved_frames >= max_frames:
            break
        
        if frame_count % skip_frames == 0:
            frame_path = os.path.join(video_output_folder, f"{video_name}_{saved_frames:03d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved_frames += 1
        
        frame_count += 1

    cap.release()
    print(f"✔ Saved {saved_frames} frames from {video_name}")

def process_dataset():
    for label in ["real", "fake"]:
        video_dir = os.path.join(VIDEO_ROOT, label)
        save_dir = os.path.join(OUTPUT_ROOT, label)

        os.makedirs(save_dir, exist_ok=True)

        videos = os.listdir(video_dir)
        print(f"\n📁 Processing {label.upper()} videos ({len(videos)})...")

        for video in tqdm(videos):
            video_path = os.path.join(video_dir, video)
            if video.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                extract_frames(video_path, save_dir, max_frames=MAX_FRAMES)

if __name__ == "__main__":
    process_dataset()
