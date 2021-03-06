项目在Pytho 3.7.5版本,pycharm下运行。

# ss摄像头矫正

### 项目总体介绍

ss摄像头有两种摄像头模式：鱼眼摄像头和方形摄像头。鱼眼摄像头呈现的画面为半球状，画面扭曲较为严重，不利于后续摄像头画面拼接等操作；方形摄像头画面较为正常，但在边缘处仍存在扭曲情况。本模块将对摄像头画面进行矫正，获得正常视频画面。

### 项目前提条件

- 需要安装的库见 ```./requirements.txt```
- 全局变量设置见 ```./ss_config.txt```

### 模块功能

- 读取训练视频流
- 根据不同的摄像头模式（鱼眼摄像头模式和正常摄像头模式）计算相机参数和失真参数
- 根据相机参数和失真参数对待矫正图片进行实时矫正，矫正方法也分为两种（opencv默认矫正模式和超参矫正模式），其中默认矫正模式矫正效果较好，但可视范围较小；超参矫正模式边缘部分略微扭曲，但可视范围较大。
- 返回被矫正后的实时视频流

### 使用方法
- ```example_1.py```   采集棋盘图片、训练、将参数存储入pickle文件、实时运行
- ```example_2.py```   直接读取SN对应的pickle文件、实时运行
- 在两个 example文件中，均有三种矫正类型：
  1.鱼眼+默认矫正模式
  2.鱼眼+超参矫正模式
  3.方形摄像头矫正
  可按需注销调试。