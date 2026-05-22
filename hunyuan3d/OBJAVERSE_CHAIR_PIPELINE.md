# Objaverse chairs -> points.npz -> Stage 1

Цель: уйти от ShapeNet `binvox 32^3` и обучать Stage 1 на настоящих Objaverse `.glb` mesh.

## 1. Kaggle settings

```text
Internet: On
Accelerator: сначала CPU/GPU достаточно для подготовки данных
```

## 2. Скачать Objaverse chair subset

```python
%cd /kaggle/working/new-model-2d-to-3d/hunyuan3d
!pip install -q objaverse

%run prepare_objaverse_chairs.py \
  --limit 500 \
  --out /kaggle/working/objaverse_chairs \
  --download-processes 8
```

Если все ок, будет:

```text
/kaggle/working/objaverse_chairs/manifest.csv
```

## 3. Превратить GLB mesh в points.npz

```python
%cd /kaggle/working/new-model-2d-to-3d/hunyuan3d
!pip install -q trimesh rtree

%run mesh_to_points_npz.py \
  --manifest /kaggle/working/objaverse_chairs/manifest.csv \
  --out /kaggle/working/objaverse_chair_points \
  --limit 500 \
  --points 65536 \
  --shape-points 8192
```

Проверка:

```python
!find /kaggle/working/objaverse_chair_points -name points.npz | head
!find /kaggle/working/objaverse_chair_points -name points.npz | wc -l
```

## 4. Обучить Stage 1 на Objaverse mesh points

```python
%cd /kaggle/working/new-model-2d-to-3d/hunyuan3d

%env STAGE1_DATA_ROOT=/kaggle/working/objaverse_chair_points
%env STAGE1_RESULTS_DIR=/kaggle/working/shape_vae_stage1_objaverse_chairs
%env RESUME_CKPT=

%env FIELD_TYPE=occupancy
%env EPOCHS=50
%env BATCH_SIZE=4
%env NUM_WORKERS=0
%env VERBOSE_BATCH=0

%env SHAPE_POINTS=4096
%env QUERY_POINTS=4096

%env LATENT_TOKENS=256
%env LATENT_DIM=384
%env FLOW_HEADS=8
%env DECODER_HIDDEN=384
%env DECODER_LAYERS=5
%env FOURIER_BANDS=8

%env LR=2e-4
%env WEIGHT_DECAY=3e-4
%env POS_WEIGHT=3.0
%env DICE_WEIGHT=0.75
%env KL_WEIGHT=1e-4
%env BEST_METRIC=iou

%run train_stage1_shape_vae.py
```

## 5. Почему это лучше Vox32

Vox32:

```text
только 32^3 grid
тонкие части стула часто потеряны
форма грубая
```

Objaverse mesh points:

```text
continuous points
near-surface samples
реальная mesh-геометрия
лучше для ножек/спинки/подлокотников
```

## 6. Важные ограничения

Objaverse шумный:

```text
часть моделей битые
часть не стулья
часть слишком сложные
часть non-watertight
```

Поэтому начинай с `--limit 500`, потом увеличивай:

```text
500 -> 1000 -> 2000
```

Если conversion слишком медленный, уменьши:

```text
--points 32768
--shape-points 4096
```
