# Pig detector training

Wait until `download_datasets.py` has finished before preparing the training data.
Run from this project directory in PowerShell:

```powershell
& "$env:USERPROFILE\.local\bin\uv.exe" run train_yolo.py --device 0
```

This installs the training dependencies, prepares all 11 datasets, and fine-tunes
`yolo26l.pt` for 100 epochs at 640 pixels. Batch size is automatically selected
for the GPU. The project selects CUDA 12.8 PyTorch wheels for RTX 50-series support.
The first run downloads packages and pretrained weights.

Preparation keeps original splits, maps pig/pigs/Pig to class 0, removes person
annotations, and retains images without pigs as negative examples. The known
single-class `pig-hjhho`, `pig-pig-pig`, and `pig-a2jtl` exports named `0` are
treated as pig. Other unrecognized
class names stop preparation for inspection. Labels must be detection boxes.
Original downloads are preserved. Images use hard links when possible, with a
copy fallback. Do not edit the prepared images, since hard links share content.
The script checks required files, but downloads must finish before running it.
Different source datasets can contain duplicate images; the original splits are
preserved and cross-dataset duplicate leakage has not been audited.

Prepare without training:

```powershell
& "$env:USERPROFILE\.local\bin\uv.exe" run train_yolo.py --prepare-only
```

Reuse prepared data (also use this when starting another training run):

```powershell
& "$env:USERPROFILE\.local\bin\uv.exe" run train_yolo.py --data datasets/combined-pigs/data.yaml --device 0 --epochs 150 --batch 8
```

Preparation refuses to overwrite an existing output directory. After a failed
preparation, choose another `--prepared` directory. `--allow-partial` explicitly
allows preparing only datasets whose data.yaml is present; use only when their
downloads have finished.

Outputs are under `runs/detect/yolo26l-pigs` (or a numbered directory if already
used). The best model is `weights/best.pt`; the resumable checkpoint is
`weights/last.pt`:

```powershell
& "$env:USERPROFILE\.local\bin\uv.exe" run train_yolo.py --resume runs/detect/yolo26l-pigs/weights/last.pt
```

Resume uses the checkpoint's saved training settings. Use `--workers 0` if Windows
data loading causes problems. If a fixed batch runs out of GPU memory, reduce it.
