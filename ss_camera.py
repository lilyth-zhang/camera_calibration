from ss_config import *
import numpy as np
import cv2
import pickle


class VideoStream(object):
    """ Receive real-time video stream according to sn number.

    Args:
        sn: identity number of the video stream

    Returns:
        frame: single frame from the video stream
        frame_id: id of each frame
    """

    def __init__(self, sn):
        self.sn = sn

    def sn2url(self):
        url = self.sn  # 默认sn号就是视频地址
        return url

    def get_frames(self):
        url = self.sn2url()
        cap = cv2.VideoCapture(url)
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if ret:
                frame_idx += 1
                yield frame, frame_idx
            else:
                break


class CalibrationTraining(object):  # 获取K和D
    """ Using chessboard pics to train the model and
        obtain camera parameters K and distortion coefficients D

    Args:
        distort_imgs: distorted chessboard pics for model training
        CB: size of chessboard
        dim: max iteration of model training
        subpix_criteria: criteria to stop finding better corners for the model
        calibration_flags: criteria to stop model training
        mode: 2 modes of video, namely fisheye mode and normal mode

    Returns:
        K: camera parameters
        D: distortion coefficients
        DIM: size of the training images
    """

    def __init__(self, distort_imgs, CB, dim, subpix_criteria, calibration_flags, mode='fisheye'):
        self.distort_imgs = distort_imgs
        self.CHECKERBOARD = CB
        self.dimension = dim
        self.subpix_criteria = subpix_criteria
        self.calibration_flags = calibration_flags
        self.video_mode = mode  # 对应两种摄像机模式

    def find_corners(self):
        objp = np.zeros((1, self.CHECKERBOARD[0] * self.CHECKERBOARD[1], 3), np.float32)
        objp[0, :, :2] = np.mgrid[0:self.CHECKERBOARD[0], 0:self.CHECKERBOARD[1]].T.reshape(-1, 2)
        _img_shape = None
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.
        to_del_files = []
        count = 1

        for img in self.distort_imgs:
            if _img_shape == None:
                _img_shape = img.shape[:2]
            else:
                assert _img_shape == img.shape[:2], "All images must share the same size."  # 不满足就丢出一个异常
            DIM = _img_shape[::-1]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, \
                                                     cv2.CALIB_CB_ADAPTIVE_THRESH + \
                                                     cv2.CALIB_CB_FAST_CHECK + \
                                                     cv2.CALIB_CB_NORMALIZE_IMAGE)
            if ret == True:
                corners2 = cv2.cornerSubPix(gray, corners, (15, 15), (-1, -1), self.subpix_criteria)
                imgpoints.append(corners2)
                objpoints.append(objp)
            else:
                #                 print("cant find corners")
                to_del_files.append(count * 20)
            count += 1
        print("Useless frames that need to be deleted:", to_del_files)

        N_OK = len(objpoints)
        print("Found %d valid images for calibration" % N_OK)

        return objpoints, imgpoints, gray.shape[::-1], DIM

    def get_KD(self):  # 视频模式
        objpoints, imgpoints, gray_shape, DIM = self.find_corners()

        if self.video_mode == "fisheye":  # 鱼眼模式
            retval, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
                objpoints,
                imgpoints,
                gray_shape,
                None,
                None,
                flags=self.calibration_flags,
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, \
                          self.dimension, 1e-6)
            )

        if self.video_mode == "normal":
            retval, K, D, rvecs, tvecs = cv2.calibrateCamera(
                objpoints,
                imgpoints,
                gray_shape,
                None,
                None)
        #         else:
        #             print("No such camera mode！")
        return K, D, DIM


class UndistortVideo(object):
    """ Receive realtime distorted frame and return undistorted one.

    Args:
        realtime_img: distorted frame from realtime video stream
        K: camera parameters
        D: distortion coefficients
        DIM: size of the training images
        calibrate_choice: normal mode only has one calibration choice while fisheye has 2 choice,
                          namely default(smaller view) choice and custom choice(broader view)
        scale: control the size of input frame (dim2) to get better view of fisheye calibration
               method, together with balance
        balance: focal parameter of the camera
        mode: 2 modes of video, namely fisheye mode and normal mode

    Returns:
        return_image: undistorted frame for the realtime video stream
    """

    def __init__(self, realtime_img, K, D, DIM, mode, calibrate_choice="custom", scale=scale, balance=balance):
        self.realtime_img = realtime_img
        self.K = K
        self.D = D
        self.DIM = DIM
        self.calibrate_choice = calibrate_choice  # 默认方法矫正or可视范围更大的矫正
        self.scale = scale  # dim2变化的比例
        self.balance = balance
        self.video_mode = mode

    def cal_map(self):
        assert self.video_mode in ["normal", "fisheye"], "No such camera mode！"
        if self.video_mode == "normal":
            h, w = self.realtime_img.shape[:2]
            new_K, roi = cv2.getOptimalNewCameraMatrix(self.K, self.D, (w, h), 1, (w, h))
            map1, map2 = cv2.initUndistortRectifyMap(self.K, self.D, np.eye(3), new_K, (w, h), 5)

        else:  # self.video_mode == "fisheye"
            assert self.calibrate_choice in ["default", "custom"], "No such calibrateion choice!"
            roi = (0, 0, self.realtime_img.shape[:2][0], self.realtime_img.shape[:2][1])
            if self.calibrate_choice == "default":
                map1, map2 = cv2.fisheye.initUndistortRectifyMap(self.K, self.D,
                                                                 np.eye(3), self.K,
                                                                 self.DIM,
                                                                 cv2.CV_16SC2)
            else:  # self.calibrate_choice == "custom"
                dim1 = self.realtime_img.shape[:2][::-1]
                assert dim1[0] / dim1[1] == self.DIM[0] / self.DIM[1], \
                    "Image to undistort needs to have same aspect ratio as the ones used in calibration"
                dim2 = tuple([int(i * self.scale) for i in dim1])
                scaled_K = self.K * dim1[0] / self.DIM[0]  # The values of K is to scale with image dimension.
                scaled_K[2][2] = 1.0
                print(self.balance)
                new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K,
                                                                               self.D, dim2, np.eye(3),
                                                                               balance=self.balance)
                print(new_K)
                map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, self.D,
                                                                 np.eye(3), new_K, dim1, cv2.CV_16SC2)
        return map1, map2, roi

    def remap_frames(self):
        map1, map2, roi = self.cal_map()
        x, y, w, h = roi
        print(x, y, w, h)
        undistorted_img = cv2.remap(self.realtime_img, map1, map2,
                                    interpolation=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT)
        _image = undistorted_img[y:y + h, x:x + w]
        remap_frame = cv2.resize(_image, self.realtime_img.shape[:2][::-1])
        yield remap_frame

