#!/usr/bin/env python
import numpy as np
import cv2
import dlib
import sys
import os.path
from os.path import basename
import argparse
 
detector = dlib.get_frontal_face_detector()
fourcc = cv2.VideoWriter_fourcc(*'MP4')

def show(name, img, val = 0):
    cv2.imshow(name, img)
    if ord('q') == cv2.waitKey(val):
        sys.exit()

def blend_with_mask_matrix(src1, src2, mask):
    res_channels = []
    for c in range(0, src1.shape[2]):
        a = src1[:, :, c]
        b = src2[:, :, c]
        m = mask[:, :, c]
        res = cv2.add(
            cv2.multiply(b, cv2.divide(np.full_like(m, 255) - m, 255.0, dtype=cv2.CV_32F), dtype=cv2.CV_32F),
            cv2.multiply(a, cv2.divide(m, 255.0, dtype=cv2.CV_32F), dtype=cv2.CV_32F),
           dtype=cv2.CV_8U)
        res_channels += [res]
    res = cv2.merge(res_channels)
    return res

def blur_image(img, mask):
    mask = cv2.GaussianBlur(mask, (args.blur_size, args.blur_size), 60)
    blurred = cv2.GaussianBlur(cv2.bitwise_and(img, mask), (args.blur_size, args.blur_size), 60)
    mask = ~mask
    not_blured = cv2.bitwise_and(img, mask)
    res =  blend_with_mask_matrix(img, blurred, mask)
    return res

def blurFace(img, rects):
    mask = np.zeros(img.shape[:3],np.uint8)
    for x, y, w, h in rects:
        sub_face = img[y:y+h, x:x+w]
        img[y:y+sub_face.shape[0], x:x+sub_face.shape[1]] = sub_face
        h1 = int(h * 0.75)
        w1 = int(w * 0.6)
        cv2.ellipse(mask, (x+w/2, y+h/2), (w1, h1), 0, 0, 360, (255,255,255), -1)
    res = blur_image(img, mask)
    return res

def detectFaceDlib(img):
    rects = []
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    dets = detector(img, 0)
    for k, d in enumerate(dets):
        x = d.left()
        y = d.top()
        w = d.right() - d.left()
        h = d.bottom() - d.top()
        rects.append((x,y,w,h))
    return rects

def parsearguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', required=True, action="store", dest="input", help="input video.")
    parser.add_argument('-o', required=True, action="store", dest="output", help="output video.")
    parser.add_argument('-g', '--debug', action="store_true", help="Run with debug mode enabled.")
    parser.add_argument('-b', '--blur_size', action="store", default=71, type=int, help="specifies bluring size.")
    parser.add_argument('--width', action="store", default=1280, type=int, help="Output video width. default is 1280.")
    parser.add_argument('--height', action="store", default=720, type=int, help="Output video height. default is 720")
    return parser.parse_args()

def checkForInputFile():
    if not os.path.exists(args.output):
        print ("Could not find the input Video. Please check whether the provided file exists. There could be typos in the path.")
        sys.exit()

def checkForOutputFile():
    i = 1
    extension = os.path.splitext(args.output)[1]
    filename = os.path.splitext(basename(args.output))[0]
    dirname = os.path.dirname((args.output)).strip()
    while os.path.exists(args.output):
        name = dirname + "/" + filename + "_" + str(i) + extension
        i += 1
        args.output = name
        print ("The file has been renamed to '" + name + "' as the provided file exists.")

if __name__ == '__main__':
   args = parsearguments()
   #checkForInputFile()
   checkForOutputFile()
   if args.blur_size % 2 == 0 :
       args.blur_size += 1
   out = cv2.VideoWriter(args.output, fourcc, 25, (args.width,args.height))
   cap = cv2.VideoCapture(args.input)

   while (True):
       ret, frame = cap.read()
       frame = cv2.resize(frame,(args.width,args.height))
       rects = detectFaceDlib(frame)
       frame = blurFace(frame, rects)
       if (args.debug):
           show("detect_faces", frame, 1)
       else:
           out.write(frame)
