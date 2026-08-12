import os
import sys

# 【强制接管环境变量】
# 获取当前 export_rknn.py 所在的文件夹路径 (即你的项目根目录)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 强行把当前目录插到 Python 搜索路径的绝对第一位！(索引为 0)
sys.path.insert(0, current_dir)

import ultralytics

print(ultralytics.__file__)

import torch

from ultralytics import YOLO


def export_to_rknn_onnx(model_path="yolo26n.pt", imgsz=640):
    """一键导出适配 rknn 的纯净版 ONNX."""
    print(f"\n[1] 正在加载模型: {model_path} ...")
    # 加载 YOLOv26 模型 (会默认读取你修改过的 head.py)
    model = YOLO(model_path)
    pytorch_model = model.model
    pytorch_model.eval()

    print("[2] 正在注入 RKNN 导出标志位 ...")
    # 引入 Detect 基类，用于严谨的类型判断
    from ultralytics.nn.modules.head import Detect

    flag_injected = False
    for m in pytorch_model.modules():
        # 只要是 Detect 的子类（包括 v10Detect, YOLOEDetect 等），统统拦截！
        if isinstance(m, Detect):
            m.export = True
            m.format = "rknn"
            flag_injected = True
            print(f" -> 成功锁定底层检测头 [{type(m).__name__}]，已注入散装 rknn 格式标志位！")

    if not flag_injected:
        print("⚠️ 警告：未找到任何 Detect 模块，拦截失败，导出的 ONNX 可能不是 9 个张量！")

    print(f"[3] 正在构造静态虚拟输入 (Batch=1, Size={imgsz}x{imgsz}) ...")
    # 强行生成一个固定的假图片，这就把 Batch Size 彻底锁死在了 1
    dummy_input = torch.randn(1, 3, imgsz, imgsz)

    output_path = model_path.replace(".pt", ".onnx")
    print("[4] 开始底层 ONNX 导出，锁定 Opset=12 ...")

    torch.onnx.export(
        pytorch_model,
        dummy_input,
        output_path,
        verbose=False,
        input_names=["images"],
        output_names=["outputs"],  # 因为输出是个 list，这里统称 outputs
        opset_version=12,  # 极其重要：RKNN-Toolkit2 对 Opset 12 兼容性最好
        do_constant_folding=True,  # 极其重要：常量折叠，提前把能算的常数算好，减轻 NPU 负担
    )

    print(f"\n✅ 导出成功！文件已保存至: {output_path}")


if __name__ == "__main__":
    # 在这里修改你的模型权重名称和想要的分辨率
    export_to_rknn_onnx(model_path="./yolo26n.pt", imgsz=640)
