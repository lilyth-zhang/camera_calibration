from ss_camera import *
import pickle


def distort_chessimgs(path):
    video = VideoStream(path)
    distort_imgs = []
    timeF = 20
    for frame, frame_idx in video.get_frames():
        if (frame_idx % timeF == 0):  # 每隔timeF帧抽取图片作为训练集
            distort_imgs.append(frame)
    return distort_imgs


def params(sn, K, D, DIM):
    print("Saving parameters...")
    path = './params/SN_%s.pickle' % sn
    f = open(path, 'wb')
    data = {'K': K,  # 放在训练后面，根据模式不同赋予不同的值
            'D': D,
            'DIM': DIM}
    pickle.dump(data, f)
    f.close()


def undistorted_video(path, K, D, DIM, mode, calibrate_choice):
    timeF = 1
    video = VideoStream(path)
    for frame, frame_idx in video.get_frames():
        if frame_idx % timeF == 0:
            uv = UndistortVideo(frame, K, D, DIM, mode, calibrate_choice)
            remap_frame = uv.remap_frames()
            for rf in remap_frame:
                # undistort_frame = cv2.cvtColor(rf, cv2.COLOR_BGR2RGB)
                rf = cv2.resize(rf,(600,600))
                cv2.imshow("capture", rf)
                cv2.waitKey(1)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    ######  1.fisheye + default ######
    # train_video = 'fisheye.mp4'
    # mode = 'fisheye'
    # calibrate_choice = 'default'
    # test_video = 'fisheye_test.mp4'

    ######  2.fisheye + custom  ######
    # train_video = 'fisheye.mp4'
    # mode = 'fisheye'
    # calibrate_choice = 'custom'
    # test_video = 'fisheye_test.mp4'

    ######  3.normal            ######
    train_video = 'cam.mp4'
    mode = 'normal'
    calibrate_choice= 'nochoice'
    test_video = 'cam_test.mp4'

    print("Getting chessboard training images...Please wait")
    distort_imgs = distort_chessimgs(train_video)
    print("Start to train the model")
    model = CalibrationTraining(distort_imgs, CHECKERBOARD, dimension, subpix_criteria, calibration_flags, mode)
    K, D, DIM = model.get_KD()
    print("DIM=" + str(DIM))
    print("K=np.array(" + str(K.tolist()) + ")")
    print("D=np.array(" + str(D.tolist()) + ")")

    sn = train_video.split('/')[-1][:-4]
    params(sn, K, D, DIM)  # 把参数保存成pickle文件

    undistorted_video(test_video, K, D, DIM, mode, calibrate_choice)