# fcos_tutorial
[基于地平线 HAT 训练与部署 FCOS 全流程](https://blog.zzsqwq.cn/posts/hat-train-fcos/) 中涉及到的一些文件：

```
.
├── README.md 
├── mscoco  # 训练 mscoco 数据集用到的文件
│   ├── config 
│   │   └── fcos_efficientnetb0_mscoco.py # 训练的配置文件
│   ├── hat
│   │   └── mscoco.py # packer 使用的 hat 部分
│   └── inference
│       ├── inference.py # 推理代码
│       ├── kite.jpg     # 推理原图
│       ├── kite_with_bad_bbox.jpg  # 错误的 bbox
│       ├── kite_with_bbox.jpg      # 原模型推理结果
│       └── kite_with_good_bbox.jpg # 正确的 bbox
├── onnx_models        # 提到的 onnx 模型
│   ├── fcos.onnx      # 官方自带的模型
│   ├── fcos_test.onnx # 自己训练得到的模型
│   └── shape.py       # 可以为 onnx 模型添加 shape 的脚本，方便 netron 查看
└── selfcoco           # 训练自己的数据集用到的文件
    ├── config
    │   └── fcos_efficientnetb0_selfcoco.py
    ├── hat
    │   └── mscoco.py
    └── inference
        └── inference.py
