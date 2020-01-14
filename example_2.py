from ss_camera import *
import pickle

def maps(sn, mode, calibrate_choice):
    path = './params/{}_{}_{}.pickle'.format(sn, mode, calibrate_choice)
    f = open(path, 'rb')
    maps = pickle.load(f)
    map1 = maps['map1']
    map2 = maps['map2']
    roi = maps['roi']
    return map1, map2, roi

def undistorted_video(path, map1, map2, roi):
    video = VideoStream(path)
    for frame, frame_idx in video.get_frames():
        uv = UndistortVideo(map1, map2, roi, frame)
        remap_frame = uv.remap_frames()
        for rf in remap_frame:
            rf = cv2.resize(rf,(600,600))
            cv2.imwrite('1.jpg', rf)
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
    train_video = 'fisheye.mp4'
    mode = 'fisheye'
    calibrate_choice = 'custom'
    test_video = 'cam2.mp4'

    ######  3.normal            ######
    # train_video = 'cam.mp4'
    # mode = 'normal'
    # calibrate_choice= 'nochoice'
    # test_video = 'cam_test.mp4'

    sn = train_video
    map1, map2, roi = maps(sn, mode, calibrate_choice)

    undistorted_video(test_video, map1, map2, roi)