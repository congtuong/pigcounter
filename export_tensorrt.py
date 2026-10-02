"""Export a trained YOLO detector to a dynamic-batch TensorRT engine."""
import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True, help='Teacher best.pt checkpoint')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=4, help='Maximum inference batch size')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--workspace', type=float, default=4, help='TensorRT workspace GiB')
    parser.add_argument('--precision', type=int, choices=(16, 32), default=16)
    args = parser.parse_args()
    if not args.model.is_file() or args.model.suffix != '.pt':
        parser.error('--model must be an existing .pt checkpoint')
    if args.batch < 1 or args.imgsz < 32 or args.imgsz % 32 or args.workspace <= 0:
        parser.error('Use a positive batch/workspace and imgsz divisible by 32')
    from ultralytics import YOLO
    import torch
    if not torch.cuda.is_available():
        parser.error('TensorRT export requires CUDA-enabled PyTorch and an NVIDIA GPU')
    model = YOLO(str(args.model.resolve()))
    if model.task != 'detect':
        parser.error('The teacher must be an object detection model')
    engine = model.export(format='engine', imgsz=args.imgsz, batch=args.batch,
                          dynamic=True, device=args.device, workspace=args.workspace,
                          quantize=args.precision, simplify=True)
    print(f'TensorRT engine: {engine}')
    print('Use the same imgsz and an inference batch no larger than the export batch.')


if __name__ == '__main__':
    main()
