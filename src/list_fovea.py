import os
import glob

fovea_dir = "/root/autodl-tmp/longfei/surgical_tracking/data/FOVEA"
files = sorted(glob.glob(os.path.join(fovea_dir, "*")))
print(f"Total files in {fovea_dir}: {len(files)}")
for f in files[:50]:
    print(os.path.basename(f))