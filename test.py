from ultralytics import YOLO
import cv2
import os
import numpy as np
import math
model = YOLO(r"runs/detect/train2/weights/best.pt")
def create_file_path(img_num, folder_num):
    if img_num<10:
        str_i = "00"+str(img_num)
    elif 9<img_num<100:
        str_i = "0"+str(img_num)
    else:
        str_i = str(img_num)
    if folder_num<10:
        folder_str = "00"+str(folder_num)
    elif 9<folder_num<100:
        folder_str = "0"+str(folder_num)
    else:
        folder_str = str(folder_num)
    img_name = r"images\\"+str(folder_num)+"\G"+folder_str+"_IMG"+str_i+".jpg"
    return img_name


def detect_corners_coords(path_to_image):
    results = model.predict(source=path_to_image, conf=0.36)
    r = results[0]
    boxes = r.boxes
    corners_coords = []
    xyxy = boxes.xyxy.cpu().numpy()
    if(len(xyxy)<4):
        print("failed to detect corners")
        return None
    # return the midpoints of the boxes of the corners
    for i in range(4):
        x1, y1, x2, y2 = xyxy[i]
        corners_coords.append([int((x1+x2)//2), int((y1+y2)//2)])
    return corners_coords


def order_corners(quad):
    pts = np.array(quad, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def warp_board(img_path, corners, scale=1.0, margin_ratio = 0.0):
    # Perspective-warp the quad to a top-down rectangle.
    ordered = order_corners(corners)
    (tl, tr, br, bl) = ordered

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)

    maxW = int(math.ceil(max(widthA, widthB) * scale * (1.0 + margin_ratio)))
    maxH = int(math.ceil(max(heightA, heightB) * scale * (1.0 + margin_ratio)))
    maxW = max(maxW, 10)
    maxH = max(maxH, 10)

    dst = np.array([
        [0, 0],
        [maxW - 1, 0],
        [maxW - 1, maxH - 1],
        [0, maxH - 1]
    ], dtype=np.float32)

    img_bgr = cv2.imread(img_path)
    M = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(img_bgr, M, (maxW, maxH))
    return warped


def main():
    img_path = create_file_path(0, 0) # Take the first photo in 0 folder-the first number is image number, second is folder number
    corners = detect_corners_coords(img_path)
    board_img = warp_board(img_path, corners)
    cv2.namedWindow("Board", cv2.WINDOW_NORMAL)
    cv2.imshow("Board", board_img)
    cv2.resizeWindow("Board", 800, 800)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
main()