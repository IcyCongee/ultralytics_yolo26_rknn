import torch
from ultralytics import YOLO
import onnx

def export_to_rknn_onnx(model_path="yolov26n.pt", imgsz=640):
    """
    一键导出适配 rknn 的纯净版 ONNX
    """
    print(f"\n[1] 正在加载模型: {model_path} ...")
    # 加载 YOLOv26 模型 (会默认读取你修改过的 head.py)
    model = YOLO(model_path)
    pytorch_model = model.model
    pytorch_model.eval()

    # ==========================================
    # 【核心黑客操作】：强行触发我们在 head.py 写的 NPU 拦截逻辑
    # 绕过官方复杂的格式白名单验证，直接在底层模块注入 flag
    # ==========================================
    print("[2] 正在注入 RKNN 导出标志位 ...")
    for m in pytorch_model.modules():
        if type(m).__name__ == "Detect":
            m.export = True       # 告诉 head 我们在导出模式
            m.format = "rknn"     # 完美契合 if self.format == 'rknn': 的拦截条件
            print(" -> 成功锁定 Detect 头的输出格式为散装 rknn 格式！")

    # ==========================================
    # 【锁定静态维度】：NPU 最喜欢的死规定
    # ==========================================
    print(f"[3] 正在构造静态虚拟输入 (Batch=1, Size={imgsz}x{imgsz}) ...")
    # 强行生成一个固定的假图片，这就把 Batch Size 彻底锁死在了 1
    dummy_input = torch.randn(1, 3, imgsz, imgsz)

    # ==========================================
    # 【执行纯净导出】：直接调用底层的 torch.onnx
    # ==========================================
    output_path = model_path.replace(".pt", "_rknn_opt.onnx")
    print(f"[4] 开始底层 ONNX 导出，锁定 Opset=12 ...")
    
    torch.onnx.export(
        pytorch_model,
        dummy_input,
        output_path,
        verbose=False,
        input_names=["images"],
        output_names=["outputs"], # 因为输出是个 list，这里统称 outputs
        opset_version=12,         # 极其重要：RKNN-Toolkit2 对 Opset 12 兼容性最好
        do_constant_folding=True, # 极其重要：常量折叠，提前把能算的常数算好，减轻 NPU 负担
    )
    
    print(f"\n✅ 导出成功！文件已保存至: {output_path}")


if __name__ == "__main__":
    # 在这里修改你的模型权重名称和想要的分辨率
    export_to_rknn_onnx(model_path="yolov26n.pt", imgsz=640)