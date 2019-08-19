import sys,os

fileDir_ann = "Annotations"
fileDir_img = "JPEGImages"

for f in os.listdir(fileDir_ann):
    f_reverse = f[::-1]
    index = len(f) - f_reverse.find(".") - 1
    img_file = fileDir_img + '\\' + f[:index] + '.jpg'
    if not os.path.isfile(img_file):
        print fileDir_ann + '\\' + f + " is deleted"
        os.remove(fileDir_ann + '\\' + f )


for f in os.listdir(fileDir_img):
    f_reverse = f[::-1]
    index = len(f) - f_reverse.find(".") - 1
    ann_file = fileDir_ann + '\\' + f[:index] + '.xml'
    if not os.path.isfile(ann_file):
        print fileDir_img + '\\' + f + " is deleted"
        os.remove(fileDir_img + '\\' + f)

print "DONE"