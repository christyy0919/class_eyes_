import os
p = "e:\课堂图片\02\tracks_by_student"
print("path repr:", repr(p))
print("exists:", os.path.exists(p))
if os.path.exists("e:\\"):
    print("e:\\ contents:")
    for name in os.listdir("e:\\"):
        print(repr(name), os.path.isdir(os.path.join("e:\\", name)))
