# DeepLearning-for-predict-deformation

## 项目目的

此项目主要用于实现在变压边力的情况下，异形盒型件的成形参数预测。

## 环境搭建

python == 3.8

tensorflow == 2.4.0

tensorflow-gpu == 2.4.0

keras == 2.4.3

opencv-python == 4.5.3.56

numpy == 1.19.5

scikit-learn == 0.24.2

matplotlib == 3.3.2

## 代码结构

DeepLearning-for-predict-deformation
├─data
├─normal_tools
├─output
│  ├─model
│  ├─pred_result
│  ├─sim_picture
│  └─visualize
├─Predict
├─Pre_deal
├─tools_for_image
├─Train
└─visualize

#### data：用于存储需要处理的csv文件以及图像文件

#### normal_tools:处理时所需函数，无需配置，可直接使用

#### output：输出文件存储位置。

​	model:用于存储神经网络训练好的模型

​	pred_result:用于存储神经网络预测的结果

​	sim_picture:用于存储处理后的可视化图片

​	visualize:用于存储可视化结果数据

#### Predict:预测函数

#### Pred_deal:预处理数据及图片函数

#### tools_for_image:处理时所需函数，无需配置，可直接使用

#### Train:训练模型

#### visuaize:可视化处理函数

## data

预处理数据较大，故未上传，链接如下。

源文件链接

https://github.com/xiaowc/Machine-learning-aluminum-hot-stamping-multiple-variable-blank-holder-force



​	

​		

​		