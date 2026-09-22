import os
root = r"c:\Users\panda\Desktop\CEW\dataset_B_FacialImages"
dirs = sorted(os.listdir(root))
print("Contents:", dirs)
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
for d in dirs:
    dp = os.path.join(root, d)
    if os.path.isdir(dp):
        imgs = [f for f in os.listdir(dp) if os.path.splitext(f)[1].lower() in IMG_EXT]
        print(f"  {d}: {len(imgs)} images")
        if imgs:
            print(f"    sample: {imgs[0]}")
