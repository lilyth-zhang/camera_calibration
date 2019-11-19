import cv2
from ss_config import *
import numpy as np
import matplotlib.pyplot

class VideoStream(object):
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
    def __init__(self,
                 distort_imgs,
                 CB,
                 dim,
                 subpix_criteria,
                 calibration_flags,
                 mode='fisheye'):
        self.distort_imgs = distort_imgs
        self.CHECKERBOARD = CB
        self.dimension = dim
        self.subpix_criteria = subpix_criteria
        self.calibration_flags = calibration_flags
        self.video_mode = mode  # 对应两种摄像机模式

    def find_corners(self):
        objp = np.zeros((1, self.CHECKERBOARD[0] * self.CHECKERBOARD[1], 3),
                        np.float32)
        objp[0, :, :2] = np.mgrid[0:self.CHECKERBOARD[0], 0:self.CHECKERBOARD[
            1]].T.reshape(-1, 2)
        _img_shape = None
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.
        to_del_files = []
        for img in self.distort_imgs:
            if _img_shape == None:
                _img_shape = img.shape[:2]
            else:
                assert _img_shape == img.shape[:
                                               2], "All images must share the same size."  # 不满足就丢出一个异常
            DIM = _img_shape[::-1]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(
                gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH +
                cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

            if ret == True:
                corners2 = cv2.cornerSubPix(
                    gray, corners, (15, 15), (-1, -1),
                    self.subpix_criteria)  # 得到更为准确的角点像素坐标
                imgpoints.append(corners2)
                objpoints.append(objp)
            else:
                print('cant find corners')
                to_del_files.append(img)

        return objpoints, imgpoints, gray.shape[::-1], DIM

    def get_KD(self):
        objpoints, imgpoints, gray_shape, DIM = self.find_corners()

        if self.video_mode == 'fisheye':  # 鱼眼模式
            retval, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
                objpoints, imgpoints, gray_shape, None, None
                #                 flags=self.calibration_flags,
                #                 criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, dimension, 1e-6)
                ##### 用cam.mp4时，加上上面两句可以正常运行，不会报错；用fisheye.mp4时加上报错，不加则D为空，但图片好像正常的####
            )

        if self.video_mode == 'normal':
            retval, K, D, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, gray_shape, None, None)
        #         else:
        #             print("No such camera mode！")
        return K, D, DIM


class UndistortVideo(object):
    def __init__(self,
                 realtime_img,
                 K,
                 D,
                 DIM,
                 mode,
                 calibrate_choice='custom',
                 scale=0.8,
                 balance=0.3):
        self.realtime_img = realtime_img
        self.K = K
        self.D = D
        self.DIM = DIM
        self.calibrate_choice = calibrate_choice  # 默认方法矫正or可视范围更大的矫正
        self.scale = scale  # dim2变化的比例
        self.balance = balance
        self.video_mode = mode

    def mapping_choice(self):
        if self.video_mode == 'normal':
            h, w = self.realtime_img.shape[:2]
            new_K, roi = cv2.getOptimalNewCameraMatrix(self.K, self.D, (w, h),
                                                       1, (w, h))
            map1, map2 = cv2.initUndistortRectifyMap(self.K, self.D, np.eye(3),
                                                     new_K, (w, h), 5)
            return map1, map2, roi
        elif self.video_mode == 'fisheye':
            roi = (0, 0, self.realtime_img.shape[:2][0],
                   self.realtime_img.shape[:2][1])
            if self.calibrate_choice == 'default':
                map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                    self.K, self.D, np.eye(3), self.K, self.DIM, cv2.CV_16SC2)

            elif self.calibrate_choice == 'custom':
                dim1 = self.realtime_img.shape[:2][::-1]
                assert dim1[0] / dim1[1] == DIM[0] / DIM[
                    1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
                dim2 = tuple([int(i * self.scale) for i in dim1])
                scaled_K = self.K * dim1[0] / self.DIM[
                    0]  # The values of K is to scale with image dimension.
                scaled_K[2][2] = 1.0
                new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                    scaled_K, self.D, dim2, np.eye(3), balance=self.balance)
                map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                    scaled_K, self.D, np.eye(3), new_K, dim1, cv2.CV_16SC2)
            else:
                print("No such calibrateion choice！")
                map1 = None
                map2 = None
            return map1, map2, roi
        else:
            print("No such camera mode！")

    def return_frames(self):
        map1, map2, roi = self.mapping_choice()
        x, y, w, h = roi
        undistorted_img = cv2.remap(
            self.realtime_img,
            map1,
            map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT)
        final_image = undistorted_img[y:y + h, x:x + w]
        yield cv2.resize(final_image, self.realtime_img.shape[:2][::-1])


if __name__ == '__main__':
    video = VideoStream('fisheye.mp4')  # 'cam.mp4'   'fisheye.mp4'
    distort_imgs = []  # 训练集
    timeF = 30
    print("Start to read chessboard training images.")
    for frame, frame_idx in video.get_frames():
        if (frame_idx % timeF == 0):  # 每隔timeF帧抽取图片作为训练集
            distort_imgs.append(frame)
#             print(frame_idx)
    print("Finish reading chessboard training images.")

    mode = 'fisheye'  # 'normal'    'fisheye'
    calibrate_choice = 'normal'  # 'normal'   'custom'
    ###### fisheye的模式对应两种矫正，default和custom ； normal模式对应一种矫正######

    model = CalibrationTraining(distort_imgs, CHECKERBOARD, dimension,
                                subpix_criteria, calibration_flags, mode)
    K, D, DIM = model.get_KD()
    print("DIM=" + str(DIM))
    print("K=np.array(" + str(K.tolist()) + ")")
    print("D=np.array(" + str(D.tolist()) + ")")

    timeF = 30
    video = VideoStream('fisheye_test.mp4')  #'cam_test.mp4' 'fisheye_test.mp4'
    for frame, frame_idx in video.get_frames():
        if (frame_idx % timeF == 0
            ) and frame is not None:  # 每隔timeF帧抽取图片作为测试集，实际运用可以不加时间间隔
            result = UndistortVideo(frame, K, D, DIM, mode, calibrate_choice)
            final_frame = result.return_frames()
            for ff in final_frame:
                undistort_frame = cv2.cvtColor(ff, cv2.COLOR_BGR2RGB)
                cv2.imshow("capture", undistort_frame)
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break
            cv2.destroyAllWindows() 
