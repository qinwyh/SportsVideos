# SportsVideos

This repository provides an example pipeline for monocular basketball scene reconstruction, resulting in 3D shooting motions and the estimated 3D trajectories of the ball.

## Example Data Source

- Example video data (`single_view.mp4`) can be downloaded here:
  [Google Drive Link](https://drive.google.com/file/d/1lGGBksdnlsLvbbDMrlzqVLa31vHjDKFi/view?usp=sharing).

- After downloading, please place `single_view.mp4` into the `input/` directory.

## Prerequisites

Ball and human detection in this project are implemented using [YOLOv5](https://github.com/ultralytics/yolov5) (AGPLv3 License).

```bash
git submodule update --init
```

Installation dependencies include:
- Python 3.8+
- PyTorch
- OpenCV

```bash
git install -r requirements.txt
```

## TODO
**This repository is under construction.**