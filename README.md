# Pig detector training

Download the datasets first with `uv run download_datasets.py` and your
`ROBOFLOW_API_KEY` environment variable. The list includes 11 datasets, including
`data-qtyxx/pig-4ri1b` and the corrected `cong-tuong/pig-mkd4a-wyabd` project.
Training currently excludes `pig-sei9k` from all splits because of noisy labels.
Its downloaded files are retained for later pseudo labeling. The root YAML
uses the other 10 datasets.

Prepare the root `data.yaml`:

```powershell
& "$env:USERPROFILE\.local\bin\uv.exe" run prepare_data.py
```

Preparation pools original train, validation, and test images across the 10 included
datasets, then shuffles them with seed 42 and creates a 90:10 train/validation
split. No test split is generated. Datasets with only test images are included
in the pool too. Use `--seed` to select another reproducible split.

Preparation writes root `data.yaml` and image-path lists under `datasets/splits`.
YOLO loads the pooled datasets through those lists; no duplicate image copies or
combined image folder are needed. It never changes images, labels, class IDs, or annotation
formats. The shared class mapping is `0: pig`, `1: person`, as requested. All
sources must already use that mapping and compatible annotations. Source
metadata declaring class 0 as person produces a warning, without remapping.
Run preparation again after adding datasets. `--allow-partial` permits missing
datasets. Duplicate/near-duplicate images across datasets have not been audited;
this random file split does not guarantee separation by camera, video, or scene.
Changing the ratio does not balance the source dataset sizes or object classes.
The new split applies to a new training run; a running trainer retains its loaded split.

Train (outdoor augmentation is enabled by default):

```powershell
& "$env:USERPROFILE\.local\bin\uv.exe" run train_yolo.py --data data.yaml --device 0 --name yolo26l-pigs-outdoor
```

Without `--data`, training uses the root data.yaml. Preparation is a separate step;
after changing the dataset list, run `uv run prepare_data.py` again.
Defaults: pretrained `yolo26l.pt`, 100 epochs, 640 pixels, automatic GPU batch
size, CUDA 12.8 PyTorch wheels for RTX 50-series support. Dependencies and model
weights download on the first run. `--epochs`, `--imgsz`, `--batch`, and `--workers`
are adjustable. Use `--workers 0` if Windows data loading causes problems.

## Outdoor augmentation

The outdoor profile samples motion blur (3–21 pixel kernels), small rotations,
shifts, zoom, shear, and perspective changes. Brightness, contrast, gamma and
color changes cover exposure differences. A weather effect is sampled on 25% of
training images: drizzle, mild fog, shadows or sun glare. Noise and compression
simulate low-light noise and compressed video. Blur is sampled on 45% of images.
YOLO adjusts boxes for geometric changes; pixel effects keep boxes unchanged.
Validation and test images stay unaugmented. Mosaic is reduced to 30% and closes
for the final 15 epochs. `--augmentation standard` uses YOLO defaults for comparison.
Evaluate on real shaky and rainy camera footage to measure improvement.

Outputs are under `runs/detect/<name>` (or a numbered directory). Best weights:
`weights/best.pt`; resumable checkpoint: `weights/last.pt`.

```powershell
& "$env:USERPROFILE\.local\bin\uv.exe" run train_yolo.py --resume runs/detect/yolo26l-pigs-outdoor/weights/last.pt
```

Resume retains saved settings. To introduce new augmentation into an older model,
start a new run with `--model <path-to-best.pt>` and `--data data.yaml`.

## Export the YOLO26x teacher and pseudo-label pig-sei9k

After training finishes, use the actual best.pt path from your YOLO26x run:

```powershell
uv run export_tensorrt.py --model runs/detect/yolo26x-pigs/weights/best.pt --device 0
uv run pseudo_label.py --model runs/detect/yolo26x-pigs/weights/best.engine --device 0
```

Export writes best.engine beside best.pt, with FP16, dynamic inputs, maximum
batch 4, image size 640, and 4 GiB workspace by default. Ultralytics needs its
ONNX and TensorRT export dependencies installed (it may attempt installation).
Build the engine on the target GPU/software environment; rebuild when those
change. Export after training to avoid competing for GPU memory. `--precision 32`
selects FP32. Match the inference image size and keep inference batch at or below
the export maximum. Dynamic batching handles the final smaller image batch.

Pseudo labeling also accepts best.pt directly if TensorRT is unavailable:

```powershell
uv run pseudo_label.py --model runs/detect/yolo26x-pigs/weights/best.pt --conf 0.7
```

The output defaults to datasets/pig-sei9k-pseudo, with its own data.yaml,
predictions.csv and summary.json. Original annotations are never read or changed.
Images are hard-linked where possible; do not edit their contents. Teacher class
names are mapped to pig=0/person=1. Confidence is excluded from training label
rows and recorded in the CSV. Predictions replace annotations in the new dataset.
Source splits are preserved, and no-confidence images are skipped by default;
use --keep-empty only after reviewing them as negatives. Existing output folders
are refused; choose a fresh --output for another pass. Failed runs can leave an
incomplete output, so choose a fresh output when retrying. Review missed objects
and false positives before adding pseudo labels to training. A high confidence
threshold can omit real objects; treat these labels as candidates for review.

The scripts do not add pseudo labels to the current root data.yaml. Keep the
original noisy dataset excluded until review is complete. Pseudo-labeled val/test
splits are not independent ground truth; use human-reviewed data for evaluation.
The teacher currently has little or no person supervision after excluding
pig-sei9k; review person predictions carefully before trusting them.
