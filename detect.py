from ultralytics import YOLO
import cv2
import os
import numpy as np
import math
model = YOLO(r"runs/detect/train5/weights/best.pt")
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


def mid_point_of_boxes(coords):
    new_coords = []
    for i in range(len(coords)):
        x1, y1, x2, y2 = coords[i]
        new_coords.append([int((x1+x2)//2), int((y1+y2)//2)])
    return new_coords


def detect_corners_and_orientation(path_to_img):
    results = model.predict(source=path_to_img, conf=0.36)
    r = results[0]
    boxes = r.boxes
    class_ids = boxes.cls.cpu().numpy()
    coords = boxes.xyxy.cpu().numpy()
    # get the boxes coordinates of the corners, the 1s and the 8s rows
    corners_coords = coords[class_ids == 0]
    eights_coords = coords[class_ids == 1]
    ones_coords = coords[class_ids == 2]
    # get the coordinates of the middle point of the boxes
    corners_coords = mid_point_of_boxes(corners_coords)
    ones_coords = mid_point_of_boxes(ones_coords)
    eights_coords = mid_point_of_boxes(eights_coords)
    return [corners_coords, ones_coords, eights_coords]


def distance(pt1, pt2):
    return (pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2

def closest_corner(corners, pt):
    min_distance = distance(pt, corners[0])
    min_distance_ind = 0
    for i in range(1, len(corners)):
        if distance(pt, corners[i])<min_distance:
            min_distance = distance(pt, corners[i])
            min_distance_ind = i
    return min_distance_ind


def order_corners_clockwise(corners):
    # This method returns the corners in a clockwise order, regardless of the 1st and 8th row
    corners.sort(key=lambda corner: corner[1])
    top = corners[:2]
    bottom = corners[2:4]
    top.sort(key = lambda corner: corner[0])
    bottom.sort(key = lambda corner: corner[0], reverse = True)
    return top + bottom


def order_corners(corners, ones, eights):
    # This method fixes the top row to be the 8th row
    if(len(corners)>=4):
        corners = corners[:4]
        corners = order_corners_clockwise(corners)
        print(f"eights length is: {len(eights)}")
        print(f"len ones is {len(ones)}")
        if(len(eights) == 2):
            tl_ind = closest_corner(corners, eights[0])
            tr_ind = closest_corner(corners, eights[1])
            if {tl_ind, tr_ind} == {0, 3}:
                first = 3
            else:
                first = min(tl_ind, tr_ind)
            # Cyclic shifting the sorted array so that first is in the beginning
            corners = corners[first:4] + corners[:first]
            return np.array(corners, dtype=np.float32).reshape(4, 2)
        elif(len(ones) == 2):
            wh_0 = closest_corner(corners, ones[0])
            wh_1 = closest_corner(corners, ones[1])
            tr_ind, tl_ind = list({0, 1, 2, 3}-{wh_0, wh_1})
            if {tl_ind, tr_ind} == {0, 3}:
                first = 3
            else:
                first = min(tl_ind, tr_ind)
            # Cyclic shifting the sorted array so that first is in the beginning
            corners = corners[first:4] + corners[:first]
            return np.array(corners, dtype=np.float32).reshape(4, 2)
        return np.array(corners, dtype = np.float32).reshape(4, 2)



def warp_quad(path_to_img, quad_pts, ones, eights, scale=1.0, margin_ratio=0):
    """
    Perspective-warp the quad to a top-down rectangle.
    Output size inferred from the quad side lengths, with optional margin.
    Returns warped
    """
    ordered = order_corners(quad_pts, ones, eights)
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

    img_bgr = cv2.imread(path_to_img)
    M = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(img_bgr, M, (maxW, maxH))
    return warped

def draw_results(path_to_img, corners, ones, eights):
    img = cv2.imread(path_to_img)
    for i in range(4):
        cv2.putText(img, str(i), (int(corners[i][0]), int(corners[i][1])), 
        cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0), 2)   
    for eight in eights:
        cv2.circle(img, eight, 15, (0, 255, 0), -1)
    for one in ones:    
        cv2.circle(img, one, 15, (0, 0, 255), -1)
    return img


def detect_and_crop():
    path_to_img = create_file_path(15, 1)
    corners, ones, eights = detect_corners_and_orientation(path_to_img)
    warped = warp_quad(path_to_img, corners, ones, eights)
    cv2.namedWindow("Board", cv2.WINDOW_NORMAL)  # or: cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO
    cv2.imshow("Board", warped)

    cv2.resizeWindow("Board", 800, 800)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test():
    path_to_img = create_file_path(1, 1)
    corners, ones, eights = detect_corners_and_orientation(path_to_img)
    corners = order_corners(corners, ones, eights)
    img = draw_results(path_to_img, corners, ones, eights)
    cv2.namedWindow("Board", cv2.WINDOW_NORMAL)  # or: cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO
    cv2.imshow("Board", img)

    cv2.resizeWindow("Board", 800, 800)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

detect_and_crop()