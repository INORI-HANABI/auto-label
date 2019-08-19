import numpy as np
import sys, os
import cv2

caffe_root = '/home/vincent/jp/caffe/'
sys.path.insert(0, caffe_root + 'python')

print("\n\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\NOTE\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\:\n")
print("the argv[1] is videos_dir and the argv[2] is output_dir,make sure that the output_dir is empty\n")
test_dir = sys.argv[1]
output_dir = sys.argv[2]

image_dir = "JPEGImages"
xml_dir = "Annotations"

import caffe

save_thre_min = 0
save_thre_max = 0.9


net_file = 'example/oepc.prototxt'
caffe_model = 'snapshot/oepc.caffemodel' 



if not os.path.exists(caffe_model):
    print("MobileNetSSD_deploy.caffemodel does not exist,")
    print("use merge_bn.py to generate it.")
    #exit()
net = caffe.Net(net_file, caffe_model, caffe.TEST)

CLASSES = ('background',
           'Head')


def preprocess(src):
    img = cv2.resize(src, (320, 240))
    img = img - 127.5
    img = img * 0.007843
    return img


def postprocess(img, out):
    h = img.shape[0]
    w = img.shape[1]
    box = out['detection_out'][0, 0, :, 3:7] * np.array([w, h, w, h])

    cls = out['detection_out'][0, 0, :, 1]
    conf = out['detection_out'][0, 0, :, 2]
    return (box.astype(np.int32), conf, cls, h, w)

def detect(videofile, f):
    cap = cv2.VideoCapture(videofile)
    index4name = f.rfind('.')
    file_name = f[:index4name]
    path_name = output_dir + '/' + file_name
    if not os.path.exists(path_name and path_name + '/' + image_dir and path_name + '/' + xml_dir):
    	os.mkdir(path_name)
    	os.mkdir(path_name + '/' + image_dir)
    	os.mkdir(path_name + '/' + xml_dir)
    count = 0
    while 1:
        ret,frame=cap.read()
        if ret==False:
            print ('error or video:' + file_name + 'is finished')
            cap.release()
            break

        origimg=frame
        img = preprocess(origimg)

        img = img.astype(np.float32)
        img = img.transpose((2, 0, 1))

        net.blobs['data'].data[...] = img
        out = net.forward()
        box, conf, cls, h, w = postprocess(origimg, out)

        save_flag = 0

        for i in range(len(box)):
            #p1 = (box[i][0], box[i][1])
            #p2 = (box[i][2], box[i][3])
            #cv2.rectangle(origimg, p1, p2, (0, 255, 0))
            #p3 = (max(p1[0], 15), max(p1[1], 15))
            #title = "%s:%.2f" % (CLASSES[int(cls[i])], conf[i])
            #cv2.putText(origimg, title, p3, cv2.FONT_ITALIC, 0.6, (0, 255, 0), 1)
            if conf[i] > save_thre_min and conf[i] < save_thre_max:
                save_flag = 1
        cv2.imshow("SSD", origimg)


        if save_flag == 1:
            count = count + 1
            save_path = path_name + '/' + image_dir + '/' + file_name + '_' + str(count) + '.jpg'
            cv2.imwrite(save_path, origimg)
            xml_file = open((path_name + '/' + xml_dir + '/' + file_name + '_' + str(count) + '.xml'), 'w')
            xml_file.write('<annotation>\n')
            xml_file.write('    <folder>VOC2007</folder>\n')
            xml_file.write('    <filename>' + file_name +  '_' + str(count)  + '.jpg' + '</filename>\n')
            xml_file.write('    <size>\n')
            xml_file.write('        <width>' + str(w) + '</width>\n')
            xml_file.write('        <height>' + str(h) + '</height>\n')
            xml_file.write('        <depth>3</depth>\n')
            xml_file.write('    </size>\n')

            for i in range(len(box)):
                p1 = (box[i][0], box[i][1])
                p2 = (box[i][2], box[i][3])
                xml_file.write('    <object>\n')
                xml_file.write('        <name>' + str(CLASSES[int(cls[i])]) + '</name>\n')
                xml_file.write('        <bndbox>\n')
                xml_file.write('            <xmin>' + str(box[i][0]) + '</xmin>\n')
                xml_file.write('            <ymin>' + str(box[i][1]) + '</ymin>\n')
                xml_file.write('            <xmax>' + str(box[i][2]) + '</xmax>\n')
                xml_file.write('            <ymax>' + str(box[i][3]) + '</ymax>\n')
                xml_file.write('        </bndbox>\n')
                xml_file.write('    </object>\n')

            xml_file.write('</annotation>')
            xml_file.close()

        key=cv2.waitKey(1)


for f in os.listdir(test_dir):
    if detect(test_dir + "/" + f,f) == False:
       break
