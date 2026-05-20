import time, os
from PIL import Image

fake_dir = r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\Data\Video\FF++C23\train\fake"
files = [f for f in os.listdir(fake_dir) if f.endswith(".mp4")]
print(f"Fake videos in train/fake: {len(files)}")
print("Sample filenames:")
for f in files[:5]:
    print(f"  {f}")


source_root = r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\datasets\ff++c23\FaceForensics++_C23"

for method in ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter", "DeepFakeDetection"]:
    folder = os.path.join(source_root, method)
    if os.path.exists(folder):
        files = [f for f in os.listdir(folder) if f.endswith(".mp4")]
        print(f"{method}: {len(files)} files")
        print(f"  sample: {files[0] if files else 'empty'}")



ffpp_cache   = r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\Data\Video\FF++C23\cached_faces\train"
celebdf_cache = r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\Data\Video\CelebDF\cached_faces\train"

for name, path in [("FF++", ffpp_cache), ("CelebDF", celebdf_cache)]:
    dirs   = os.listdir(path)
    # check a sample entry has 24 frames
    sample = os.path.join(path, dirs[0])
    frames = [f for f in os.listdir(sample) if f.endswith(".png")]
    print(f"{name}: {len(dirs)} cached videos  |  sample frames: {len(frames)}")

cache = r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\Data\Video\CelebDF\cached_faces\train"
sample_dir = os.path.join(cache, os.listdir(cache)[0])
pngs = sorted([f for f in os.listdir(sample_dir) if f.endswith(".png")])

# time raw PNG reads
t = time.time()
for f in pngs:
    img = Image.open(os.path.join(sample_dir, f)).convert("RGB")
print(f"24 PNG reads: {(time.time()-t)*1000:.1f}ms")

# time transform too
from torchvision import transforms
tf = transforms.Compose([transforms.ToTensor(),
     transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

t = time.time()
for f in pngs:
    img = Image.open(os.path.join(sample_dir, f)).convert("RGB")
    tf(img)
print(f"24 PNG reads + transform: {(time.time()-t)*1000:.1f}ms")

