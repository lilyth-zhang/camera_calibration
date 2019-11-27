from ss_camera import *
import pickle

def K_D_DIM(sn):
    path = './params/SN_%s.pickle' % sn
    f = open(path, 'rb')
    k_d_dim = pickle.load(f)
    K = k_d_dim['K']
    D = k_d_dim['D']
    DIM = k_d_dim['DIM']
    return K, D, DIM

def undistorted_video(path, K, D, DIM, mode, calibrate_choice):
    timeF = 1
    video = VideoStream(path)
    for frame, frame_idx in video.get_frames():
        if frame_idx % timeF == 0:
            uv = UndistortVideo(frame, K, D, DIM, mode, calibrate_choice)
            remap_frame = uv.remap_frames()
            for rf in remap_frame:
                # undistort_frame = cv2.cvtColor(rf, cv2.COLOR_BGR2RGB)
                cv2.imshow("capture", rf)
                cv2.waitKey(10)
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
    test_video = 'fisheye_test.mp4'

    ######  3.normal            ######
    # train_video = 'cam.mp4'
    # mode = 'normal'
    # calibrate_choice= 'nochoice'
    # test_video = 'cam_test.mp4'

    sn = train_video.split('/')[-1][:-4]
    K, D, DIM = K_D_DIM(sn)  # 读取pickle获取K,D,DIM

    undistorted_video(test_video, K, D, DIM, mode, calibrate_choice)