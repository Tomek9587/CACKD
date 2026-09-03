import os
import cv2

ann_path = "/root/autodl-tmp/longfei/surgical_tracking/data/MICCAI14_Yeqing-master/MICCAI14_Yeqing-master/RetinalData/package/annotation_source/test1.txt"
frames_dir = "/root/autodl-tmp/longfei/surgical_tracking/data/MICCAI14_Yeqing-master/MICCAI14_Yeqing-master/RetinalData/separate/seq1"
out_dir = "/root/autodl-tmp/longfei/surgical_tracking/data/inspect_tip_outputs"
max_frames = 50
src_w = 1920
src_h = 1080
enable_scale = False
use_tip_only = False
# demo_groundtruth.m uses truth(:, [5 4]) => take the 2nd point and swap x/y
tip_point_index = 1
tip_swap_xy = True
variants = [
    ("base", False, False, False),
    ("flipx", False, True, False),
    ("flipy", False, False, True),
    ("swap", True, False, False),
    ("swap_flipx", True, True, False),
    ("swap_flipy", True, False, True)
]

colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0)]

with open(ann_path, "r", encoding="utf-8") as f:
    lines = [l.strip() for l in f if l.strip()]

os.makedirs(out_dir, exist_ok=True)

count = 0
for i, line in enumerate(lines):
    if max_frames is not None and count >= max_frames:
        break
    parts = line.split()
    frame_id = parts[0]
    nums = list(map(float, parts[1:]))
    pairs = [(nums[j], nums[j+1]) for j in range(0, len(nums), 2)]
    if use_tip_only:
        if tip_point_index >= len(pairs):
            continue
        tx, ty = pairs[tip_point_index]
        if tip_swap_xy:
            tx, ty = ty, tx
        pairs = [(tx, ty)]

    frame_file = os.path.join(frames_dir, f"{frame_id}.png")
    if not os.path.isfile(frame_file):
        continue

    img = cv2.imread(frame_file)
    if img is None:
        continue
    for name, swap_xy, flip_x, flip_y in variants:
        img_v = img.copy()
        for k, (x, y) in enumerate(pairs):
            if swap_xy:
                x, y = y, x
            if enable_scale and src_w and src_h:
                x = x * (img_v.shape[1] / float(src_w))
                y = y * (img_v.shape[0] / float(src_h))
            if flip_x:
                x = (img_v.shape[1] - 1) - x
            if flip_y:
                y = (img_v.shape[0] - 1) - y
            cv2.circle(img_v, (int(x), int(y)), 4, colors[k], -1)
            cv2.putText(img_v, str(k), (int(x)+5, int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[k], 1)
        out_sub = os.path.join(out_dir, name)
        os.makedirs(out_sub, exist_ok=True)
        out_path = os.path.join(out_sub, f"{frame_id}.png")
        cv2.imwrite(out_path, img_v)
    count += 1
