from ss_camera import *
import pickle


def distort_chessimgs(path):
    video = VideoStream(path)
    distort_imgs = []
    for frame, frame_idx in video.get_frames():
        if (frame_idx % timeF == 0):  # 每隔timeF帧抽取图片作为训练集
            distort_imgs.append(frame)
    print("Obtain %d chessboard images for training." % len(distort_imgs))
    return distort_imgs

def params(sn, mode, calibrate_choice, map1, map2, roi):
    print("Saving parameters...")
    path = './params/{}_{}_{}.pickle'.format(sn, mode, calibrate_choice)
    f = open(path, 'wb')
    data = {'map1': map1,  # 放在训练后面，根据模式不同赋予不同的值
            'map2': map2,
            'roi': roi}
    pickle.dump(data, f)
    f.close()

def undistorted_video(path, map1, map2, roi):
    video = VideoStream(path)
    for frame, frame_idx in video.get_frames():
        uv = UndistortVideo(map1, map2, roi, frame)
        remap_frame = uv.remap_frames()
        for rf in remap_frame:
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
    # size_test_image = (1440,1440)

    ######  2.fisheye + custom  ######
    train_video = 'fisheye.mp4'
    mode = 'fisheye'
    calibrate_choice = 'custom'
    test_video = 'cam2.mp4'
    size_test_image = (1440,1440)

    ######  3.normal            ######
    # train_video = 'cam.mp4'
    # mode = 'normal'
    # calibrate_choice= 'nochoice'
    # test_video = 'cam_test.mp4'
    # size_test_image = (1440,1440)

    print("Stage1: Extract chessboard images from %s..." % train_video)
    distort_imgs = distort_chessimgs(train_video)
    print("Stage2: Start training the model...")
    model = CalibrationTraining(distort_imgs, mode, calibrate_choice, size_test_image)
    map1, map2, roi = model.cal_map()

    sn = train_video
    params(sn, mode, calibrate_choice, map1, map2, roi)  # 把参数保存成pickle文件

    undistorted_video(test_video, map1, map2, roi)