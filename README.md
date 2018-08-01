# Compress-Neural-Networks-With-Dynamic-Quantization
一种基于动态量化编码的深度神经网络压缩方法

   将深度神经网络中的权重量化为2的n（n为整数）次幂  
优先量化权重绝对值大的值，那些值较小的值量化码表的下限或者0。
采用动态量化编码的方法，码表中没有0的结果：

![image](https://github.com/chuanraoCV/Compress-Neural-Networks-With-Dynamic-Quantization/tree/master/reasult/结果.png)

码表中有0，量化的结果：

![image](https://github.com/chuanraoCV/Compress-Neural-Networks-With-Dynamic-Quantization/tree/master/reasult/结果1.png)


