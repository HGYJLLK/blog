
+++
title = '如何部署OpenCV在ESP32中的手势识别项目'
date = 2024-07-08T21:00:56+08:00
draft = true
+++

根据作者的 README.md 可得知，git 仓库中没有数据集，需要自己下载。

* 该项目用到的制作数据集下载地址：
  * [百度网盘](https://pan.baidu.com/s/1KY7lAFXBTfrFHlApxTY8NA)（密码: ara8）

下载后可以看到有两个之前报错的点，因为缺少 handpose_datasets_v2/v1 的数据集。将这个数据集放到根目录下即可正常运行。运行方式是在打包的 zip 项目下的 ./image 文件夹中，有照片识别这些照片手指的位置。

至此已完成引用数据集。项目链接：
[https://github.com/EricLee2021-72324/handpose_x](https://github.com/EricLee2021-72324/handpose_x)

## 手势识别

### 环境要求：
- Python 3.7
- PyTorch >= 1.5.1
- OpenCV-Python

### Windows 安装步骤：

1. **安装 Anaconda**

   打开 [anaconda.com](https://www.anaconda.com)，下载并安装后，配置环境变量。在 bin 中找到 conda.exe，然后将 bin 的路径复制到环境变量的 PATH 中。打开 cmd 输入：
   ```bash
   conda -V
   ```
   验证 Anaconda 安装成功。

2. **安装 CUDA**

   按照开源文件所示需要安装 CUDA。安装 10.2 版本的 CUDA（因为文件中需要 1.5.1 版本的 PyTorch）。安装完成后输入：
   ```bash
   nvcc -V
   ```
   验证 CUDA 安装成功。

3. **部署 Conda 环境**

   按照开源文件显示需要 Python 3.7 的环境，在 cmd 中创建环境：
   ```bash
   conda create -n py37 python=3.7
   ```
   激活并使用环境：
   ```bash
   conda activate py37
   ```
   如果经常忘记 `activate` 怎么拼写，使用 `conda activate py37` 打开帮助，第一行就是这个单词。

4. **安装 PyTorch**

   按照作者要求安装 1.5.1 版本的 PyTorch：
   ```bash
   pip install torch===1.5.1 torchvision===0.6.1 -f https://download.pytorch.org/whl/torch_stable.html -i https://pypi.douban.com/simple
   ```
   或者使用 Conda 安装：
   ```bash
   conda install pytorch torchvision cudatoolkit=10.2
   ```

5. **安装 OpenCV-Python**

   一行命令安装：
   ```bash
   pip install opencv-python
   ```
   下载完成后用以下命令检查安装是否成功：
   ```python
   import cv2
   ```

6. **从 GitHub 下载项目**

   下载 zip 压缩包，解压后打开 cmd：
   ```bash
   cd 安装位置
   ```
   至此环境已经安装完毕。

### 实现手势识别

使用以下 Python 代码实现手势识别：

```python
import os
import sys
import cv2
import numpy as np
import onnxruntime
from hand_data_iter.datasets import draw_bd_handpose

class ONNXModel:
    def __init__(self, onnx_path, gpu_cfg=False):
        self.onnx_session = onnxruntime.InferenceSession(onnx_path)
        if gpu_cfg:
            self.onnx_session.set_providers(['CUDAExecutionProvider'], [{'device_id': 0}])
        self.input_name = self.get_input_name(self.onnx_session)
        self.output_name = self.get_output_name(self.onnx_session)

    def get_output_name(self, onnx_session):
        output_name = []
        for node in onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    def get_input_name(self, onnx_session):
        input_name = []
        for node in onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def get_input_feed(self, input_name, image_numpy):
        input_feed = {}
        for name in input_name:
            input_feed[name] = image_numpy
        return input_feed

    def forward(self, image_numpy):
        input_feed = self.get_input_feed(self.input_name, image_numpy)
        output = self.onnx_session.run(self.output_name, input_feed=input_feed)
        return output

if __name__ == "__main__":
    img_size = 256
    model = ONNXModel("resnet_50_size-256.onnx")

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_width = frame.shape[1]
        img_height = frame.shape[0]
        img = cv2.resize(frame, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
        img_ndarray = img.transpose((2, 0, 1))
        img_ndarray = img_ndarray / 255.0
        img_ndarray = np.expand_dims(img_ndarray, 0)

        output = model.forward(img_ndarray.astype('float32'))[0][0]
        output = np.array(output)

        pts_hand = {}
        for i in range(int(output.shape[0] / 2)):
            x = (output[i * 2 + 0] * float(img_width))
            y = (output[i * 2 + 1] * float(img_height))
            pts_hand[str(i)] = {"x": x, "y": y}

        draw_bd_handpose(frame, pts_hand, 0, 0)

        for i in range(int(output.shape[0] / 2)):
            x = (output[i * 2 + 0] * float(img_width))
            y = (output[i * 2 + 1] * float(img_height))
            cv2.circle(frame, (int(x), int(y)), 3, (255, 50, 60), -1)
            cv2.circle(frame, (int(x), int(y)), 1, (255, 150, 180), -1)

        cv2.imshow('image', frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
```

以上代码实现了摄像头捕捉手部图像并使用 ONNX 模型进行手势识别。但存在以下问题：
* 只能识别一只手，无法识别两只手或左右手。
* 当画面没有手时会乱识别。
* 当手在摄像头拍摄范围边缘时，手掌位置识别错误。
* 手掌方向识别良好，但手背识别较差。

