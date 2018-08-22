### Title：使用TensorFlow进行图像样本的预处理，包括图像的裁剪、大小缩放、图像翻转
目的：此repo的目的是对图像训练样本进行数据增强（Data Augmentation),对图像进行裁剪、大小缩放、图像翻转、图像色彩、明暗、对比度的变化。这样就实现了人为的数据扩充，使训练的模型更具鲁棒性，能应对更复杂情况下图像的预测。

主要使用的库、API：

```
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
```
其中TensorFlow的图像处理函数是：

1.tf.gfile.FastGFile  读取一张图像  
2.tf.image.decode_jpeg  解码jpg格式的图像，转换为TensorFlow可以处理的数据格式
3.tf.image.random_brightness ###调整图像的亮度  
tf.image.random_saturation ##调整图像的饱和度  
tf.image.random_hue  #调整图像的色调
tf.image.random_contrast  ##调整图像的对比度  
#### def distort_color
这个函数是对图像进行处理预处理，随机调整图像的色彩、调整亮度、对比度、饱和度、色相等图像的属性，会影响最后的结果，所以对图像预处理，把这些变化加入进去，这样可以进一步降低无关因素对模型的影响  
### def preprocess_for_train
这个函数是给定一张解码后的图像，目标图像的尺寸以及图像上的标注框，此函数可以对给出的图像进行预处理。这个函数的输入图像是图像识别问题中的
原始训练图像，而输出则是神经网络的输入层。这里只处理模型的训练数据，对于预测的数据，一般不需要随机变换的步骤  
with tf.Session() as sess:  相当于main函数，整个程序的入口，实现操作有：读取图像，输入处理框，显示处理的图像，把处理的图像保存到路径上，其中，显示和保存图像是matplotlib中的plt.imshow, plt.savefig, plt,show

### 使用方法：
只需把image_raw_data = tf.gfile.FastGFile(r"D:\1.jpg", 'rb').read()中的指定图片路径更改就可以了，还有确定plt.savefig(r'D:\Python\TensorFlow图像预处理图片\test%d.jpg' % i)保存处理图像的路径

