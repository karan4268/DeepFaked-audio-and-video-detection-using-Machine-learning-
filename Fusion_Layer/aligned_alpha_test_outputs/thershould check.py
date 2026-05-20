import numpy as np

audio = r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\Fusion_Layer\aligned_alpha_test_outputs\aligned_audio_scores.npy"
video = r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\Fusion_Layer\aligned_alpha_test_outputs\aligned_video_scores.npy"
labels_ = r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\Fusion_Layer\aligned_alpha_test_outputs\aligned_labels.npy"

fused = 0.45 * np.load(audio) + \
        0.55 * np.load(video)
labels = np.load(labels_)

in_band = (fused >= 0.455) & (fused < 0.55)
print(f"Samples in band      : {in_band.sum()}")
print(f"  Real in band (FP risk removed) : {((labels==0) & in_band).sum()}")
print(f"  Fake in band (FN risk added)   : {((labels==1) & in_band).sum()}")