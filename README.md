# EmoCap

参考了[**SemStyle**](https://arxiv.org/abs/1805.07030)的框架，基于源码改进得到的EmoCap。

#### 数据集

需要根据一定的规则将文本映射至义原序列，COCO数据集部分是现成的，Sentiment140部分是`./code/token_processing.py`生成的。

- Image Encoder：根据MSCOCO中的图像字幕生成的义原数据集，约12w张图片和60w条样本
- Language Generator：Sentiment140数据集 + COCO抽样，得到约58w条文本和对应的义原序列（`./code/token_processing.py`对文本进行处理得到义原序列，可以自行更改相应数据集，本项目不提供）
- Emotion Decoder：和Language Generator同样的文本数据，标签为对应的风格类型

#### 使用方法

`img_to_text.py`整合了三个模块，单模块内产生的是义原序列

`seq2seq_pytorch.py`由义原序列产生对应的三种风格化文本输出

`seq2emo.py`由风格化图像字幕产生得到对应的PAD值

运行如下命令：

`python code/img_to_text.py --test_folder <folder with test images>`

可以得到每张图像对应的义原序列、三种风格化图像字幕及对应的PAD值，同时结果将保存在对应文件夹的`result.json`中

`python code/port.py`可以在服务器上运行

#### 模型

具体的模型`./code/models`以及数据集`./datasets`，在[此处](https://drive.google.com/drive/folders/1BzHF0x_JfG6xRV4vW9z0HGKa1QbMgwhR?usp=sharing)下载

### 9/30/2023更新

增加了四篇相关的参考文献，其中Semstyle是本实验代码的基本框架，PAD的两篇文献仅作了解即可
