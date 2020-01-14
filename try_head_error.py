from ss_config import *
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle


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

        for img in self.distort_imgs:
            if _img_shape == None:
                _img_shape = img.shape[:2]
            else:
                assert _img_shape == img.shape[:2], "All images must share the same size."  # 不满足就丢出一个异常
            DIM = _img_shape[::-1]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, \
                                                     cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
            if ret == True:
                corners2 = cv2.cornerSubPix(gray, corners, (15, 15), (-1, -1), self.subpix_criteria)
                imgpoints.append(corners2)
                objpoints.append(objp)
            else:
                print("cant find corners")
        #                 to_del_files.append(count*20)
        #         print(to_del_files)

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
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, self.dimension, 1e-6)
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
    def __init__(self, realtime_img, K, D, DIM, mode, calibrate_choice, scale=0.8, balance=0.3):
        self.realtime_img = realtime_img
        self.K = K
        self.D = D
        self.DIM = DIM
        self.calibrate_choice = calibrate_choice  # 默认方法矫正or可视范围更大的矫正
        self.scale = scale  # dim2变化的比例
        self.balance = balance
        self.video_mode = mode

    def mapping_choice(self):
        if self.video_mode == "normal":
            h, w = self.realtime_img.shape[:2]
            print(h,w)
            new_K, roi = cv2.getOptimalNewCameraMatrix(self.K, self.D, (w, h), 1, (w, h))
            map1, map2 = cv2.initUndistortRectifyMap(self.K, self.D, np.eye(3), new_K, (w, h), 5)
            return map1, map2, roi

        elif self.video_mode == "fisheye":
            roi = (0, 0, self.realtime_img.shape[:2][0], self.realtime_img.shape[:2][1])
            if self.calibrate_choice == "default":
                map1, map2 = cv2.fisheye.initUndistortRectifyMap(self.K, self.D,
                                                                 np.eye(3), self.K, self.DIM, cv2.CV_16SC2)
            elif self.calibrate_choice == "custom":
                dim1 = self.realtime_img.shape[:2][::-1]
                assert dim1[0] / dim1[1] == self.DIM[0] / self.DIM[
                    1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
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
        print(x, y, w, h)
        undistorted_img = cv2.remap(self.realtime_img, map1, map2,
                                    interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        _image = undistorted_img[y:y + h, x:x + w]
        return_image = cv2.resize(_image, self.realtime_img.shape[:2][::-1])
        yield return_image


def distort_chessimgs(path):
    video = VideoStream(path)
    distort_imgs = []
    timeF = 20
    for frame, frame_idx in video.get_frames():
        if (frame_idx % timeF == 0):  # 每隔timeF帧抽取图片作为训练集
            distort_imgs.append(frame)
    return distort_imgs


def undistorted_video(path, K, D, DIM, mode, calibrate_choice):
    frame = cv2.imread(path)
    frame = cv2.resize(frame, (1440,1440))
    result = UndistortVideo(frame, K, D, DIM, mode, calibrate_choice)
    result_frame = result.return_frames()
    for rf in result_frame:
        # plt.imshow(rf)
        # plt.show()
        # undistort_frame = cv2.cvtColor(rf, cv2.COLOR_BGR2RGB)
        cv2.imshow("capture", rf)
        cv2.waitKey(0)
    cv2.destroyAllWindows()


def params(sn, K, D, DIM):
    print("Saving parameters...")
    path = './params/SN%s.json' % sn
    f = open(path, 'wb')
    data = {'K': np.asarray(K).tolist(),  # 放在训练后面，根据模式不同赋予不同的值
            'D': np.asarray(D).tolist(),
            'DIM': DIM}
    pickle.dump(data, f)
    f.close()


def K_D_DIM(sn):
    path = './params/SN%s.json' % sn
    f = open(path, 'rb')
    k_d_dim = pickle.load(f)
    return k_d_dim


if __name__ == '__main__':
    ######  1.fisheye + default ######
    #     train_video = 'fisheye.mp4'
    #     mode = 'fisheye'
    #     calibrate_choice= 'default'
    #     test_video = 'fisheye_test.mp4'

    ######  2.fisheye + custom  ######
    #     train_video = 'fisheye.mp4'
    #     mode = 'fisheye'
    #     calibrate_choice = 'custom'
    #     test_video = 'fisheye_test.mp4'

    ######  3.normal            ######
    train_video = 'cam.mp4'
    mode = 'normal'
    calibrate_choice = 'nochoice'
    # test_video = 'cam_test.mp4'

    #     print("Getting chessboard training images...Please wait")
    #     distort_imgs = distort_chessimgs(train_video)
    #     print("Training the model...")
    #     model = CalibrationTraining(distort_imgs, CHECKERBOARD, dimension, subpix_criteria, calibration_flags, mode)
    #     K,D,DIM = model.get_KD()
    #     print("DIM=" + str(DIM))
    #     print("K=np.array(" + str(K.tolist()) + ")")
    #     print("D=np.array(" + str(D.tolist()) + ")")

    sn = train_video.split('/')[-1][:-4]
    #     params(sn,K,D,DIM)

    k_d_dim = K_D_DIM(sn)
    K = np.array(k_d_dim['K'])
    D = np.array(k_d_dim['D'])
    DIM = k_d_dim['DIM']
    print(type(K))

    test_video = '0_12_vis.jpg'

    undistorted_video(test_video, K, D, DIM, mode, calibrate_choice)