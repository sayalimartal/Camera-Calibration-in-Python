###############
##Design the function "calibrate" to  return 
# (1) intrinsic_params: should be a list with four elements: [f_x, f_y, o_x, o_y], where f_x and f_y is focal length, o_x and o_y is offset;
# (2) is_constant: should be bool data type. False if the intrinsic parameters differed from world coordinates. 
#                                            True if the intrinsic parameters are invariable.
#It is ok to add other functions if you need
###############
import numpy as np
from cv2 import imread, cvtColor, COLOR_BGR2GRAY, TERM_CRITERIA_EPS, TERM_CRITERIA_MAX_ITER, \
    findChessboardCorners, cornerSubPix, drawChessboardCorners

def calibrate(imgname):
    img = imread(imgname)
    gray_img = cvtColor(img, COLOR_BGR2GRAY)
    retval, corners = findChessboardCorners(gray_img, (4, 9), None)

    p = 36  # 36 points on the checkerboard
    if retval:
        cornerSubPix(gray_img, corners, (11, 11), (-1, -1), (TERM_CRITERIA_EPS + TERM_CRITERIA_MAX_ITER, 30, 0.001))
        drawChessboardCorners(img, (4, 9), corners, retval)

    corners = corners.reshape(36, 2)
    world_coord =[[40,0,40],[40,0,30],[40,0,20],[40,0,10],[30,0,40],[30,0,30],[30,0,20],[30,0,10],[20,0,40],[20,0,30],[20,0,20],[20,0,10],[10,0,40],[10,0,30],[10,0,20],[10,0,10],[0,0,40],[0,0,30],[0,0,20],[0,0,10],[0,10,40],[0,10,30],[0,10,20],[0,10,10],[0,20,40],[0,20,30],[0,20,20],[0,20,10],[0,30,40],[0,30,30],[0,30,20],[0,30,10],[0,40,40],[0,40,30],[0,40,20],[0,40,10]]
    A = np.zeros((2 * p, 12), dtype=np.float64)

    for i in range(p):
        x, y = corners[i]  # image co-ordinates
        X, Y, Z = world_coord[i]  # world co-ordinates
        r1 = np.array([X, Y, Z, 1, 0, 0, 0, 0, -x * X, -x * Y, -x * Z, -x])
        r2 = np.array([0, 0, 0, 0, X, Y, Z, 1, -y * X, -y * Y, -y * Z, -y])
        A[2 * i] = r1
        A[1 + (2 * i)] = r2

    u, s, vh = np.linalg.svd(A, full_matrices=True)

    X = vh[-1]  # X is last row of vh
    X = X.reshape(3, 4)  # Converting 12*1 to 3*4
    X_norm = X[2, 0:3]

    normval = np.linalg.norm(X_norm)
    l = 1/(normval)

    M = np.dot(l, X)

    M1 = M[0, 0:3]
    M2 = M[1, 0:3]
    M3 =M[2, 0:3]

    ox = np.dot(M1.transpose(), M3)
    oy = np.dot(M2.transpose(), M3)
    fx = np.sqrt((np.dot(M1.transpose(), M1))-(ox ** 2))
    fy = np.sqrt((np.dot(M2.transpose(), M2)) - (oy ** 2))

    return [fx, fy, ox, oy], True

if __name__ == "__main__":
    intrinsic_params, is_constant = calibrate('checkboard.png')
    print(intrinsic_params)
    print(is_constant)
