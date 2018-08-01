# Compress-Neural-Networks-With-Dynamic-Quantization
一种基于动态量化编码的深度神经网络压缩方法

    采用参数共享的方式，对深度神经网络进行压缩。网络中的权重均量化2的n（n为整数）次幂这种形式。
将权重量化为2的n次幂这种新式，在进行计算时，只需一位操作可以加快运算速度。
同时，可以证明权重的绝对值越大，这种量化方式量化误差越大。
优先量化权重绝对值大的值，那些值较小的值量化码表的下限或者0。

采用动态量化编码的方法，码表中没有0的结果：
![](https://github.com/chuanraoCV/Compress-Neural-Networks-With-Dynamic-Quantization/tree/master/reasult/结果.png)


