# only to verify if any video has no usable frames / currupted
import cv2, os

data_dir = r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\Data\Video"
bad_videos = []

for split in ["train","val","test"]:
    for label in ["real","fake"]:
        folder = os.path.join(data_dir, split, label)
        for f in os.listdir(folder):
            path = os.path.join(folder, f)
            cap = cv2.VideoCapture(path)
            ret, _ = cap.read()
            cap.release()
            if not ret:
                bad_videos.append(path)

print("Videos with no usable frames:")
for v in bad_videos:
    print(v)
print(f"Total bad videos: {len(bad_videos)}")