### config ###
import cv2
CHECKERBOARD = (6,9)   # 棋盘格的大小
dimension = 25    # 最大迭代次数
subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, dimension, 0.1)
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW
timeF = 20   # 以timeF帧为间隔获取训练视频的棋盘图片
########## 控制 fisheye+custom 矫正模式的超参，详细说明见  #############
########## https://medium.com/@kennethjiang/calibrate-fisheye-lens-using-opencv-333b05afa0b0
scale=0.7
balance=0.2