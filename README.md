# Compress-Neural-Networks-With-Dynamic-Quantization
一种基于动态量化编码的深度神经网络压缩方法

将深度神经网络中的权重量化为2的n（n为整数）次幂，采用这种量化的方式为了在计算时，2的指数形式方便一位运算。同时还可以证明，将权重量化2的n次幂这种形式，权重的绝对值越大，量化误差也越大。

优先量化权重绝对值大的值，那些值较小的值量化码表的下限或者0。

采用动态量化编码的方法，码表中没有0的结果：

![](/reasult/结果.png)

码表中有0，量化的结果：

![](/reasult/结果1.png)


码表中无0的量化方式：

![](/reasult/公式1.png)

码表中有0的量化方式：

![](/reasult/公式.png)

运行方式：

（1）下载cifar-10数据集：http://www.cs.toronto.edu/~kriz/cifar.html

（2）使用data/data_tfrecord.py生成tfrecord文件

（3）使用full_precesion文件下不同深度的ResNet对网络训练，生成baseline，没有压缩前的实验结果。

（4）使用上一步生成预训练模型作为初始化，进行压缩训练：
  
   a.在code文件下有两个文件夹分别为dqc和dqc_0，代表码表中有0和没0（引入0会起到正则化的作用，但0不能表示2的n次幂这种形式，需要独立的位来表示）
   
   b.当码表中无0时，运行 python resnet_3.py --residual_net_n 3 
   上面的resnet_3.py代表将网络量化为3位，参数residual_net_n为3代表ResNet-20，若这个参数5时代表网络为ResNet-32，计算方式为6*residual_net_n+2
   
   对dqc_0中文件也是这样运行
   





