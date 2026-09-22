import os
root = r"c:\Users\panda\Desktop\CEW\dataset_B_FacialImages"
for sub in ["ClosedFace", "OpenFace"]:
    dp = os.path.join(root, sub)
    print(f"{sub}: exists={os.path.isdir(dp)}")
    if os.path.isdir(dp):
        files = os.listdir(dp)[:5]
        print(f"  first 5: {files}")
        for f in files:
            print(f"    ext: '{os.path.splitext(f)[1]}'")
