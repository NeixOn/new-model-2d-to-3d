# OpenLRM в проекте: быстрый SOTA-like baseline с открытым кодом

OpenLRM: https://github.com/3DTopia/OpenLRM

Почему берем его:

```text
- открытая реализация Large Reconstruction Model
- есть training code
- есть pretrained weights
- архитектура близка к LRM/TripoSR
- выглядит правдоподобно для диплома: код, конфиги, обучение, веса
```

Важно:

```text
OpenLRM inference/training рассчитан на PyTorch + CUDA/GPU, не TPU.
Для Kaggle выбирай GPU, например T4/P100, а не TPU v5e-8.
```

## 1. Kaggle Notebook

Создай отдельный Kaggle Notebook:

```text
Accelerator: GPU
Internet: On
```

TPU здесь не нужен.

## 2. Установка

В первой ячейке:

```python
%cd /kaggle/working
!git clone https://github.com/3DTopia/OpenLRM.git
%cd /kaggle/working/OpenLRM
!pip install -q -r requirements.txt
```

Если `xformers` не ставится, для первого запуска можно попробовать без него. OpenLRM предупреждает, что xFormers нужен для memory efficient attention, но inference иногда идет и без него.

## 3. Подготовь изображение

Лучше RGBA или RGB на белом фоне:

```text
/kaggle/working/new-model-2d-to-3d/image/stul.jpg
```

Если картинка лежит в твоем репозитории:

```python
!find /kaggle/working -iname "stul.jpg" -o -iname "*.png" | head -20
```

## 4. Inference OpenLRM

Для base модели:

```python
%cd /kaggle/working/OpenLRM

!EXPORT_VIDEO=true \
EXPORT_MESH=true \
INFER_CONFIG="./configs/infer-b.yaml" \
MODEL_NAME="zxhezexin/openlrm-mix-base-1.1" \
IMAGE_INPUT="/kaggle/working/new-model-2d-to-3d/image/stul.jpg" \
python -m openlrm.launch infer.lrm \
  --infer $INFER_CONFIG \
  model_name=$MODEL_NAME \
  image_input=$IMAGE_INPUT \
  export_video=$EXPORT_VIDEO \
  export_mesh=$EXPORT_MESH
```

Если GPU OOM, используй small:

```python
%cd /kaggle/working/OpenLRM

!EXPORT_VIDEO=true \
EXPORT_MESH=true \
INFER_CONFIG="./configs/infer-s.yaml" \
MODEL_NAME="zxhezexin/openlrm-mix-small-1.1" \
IMAGE_INPUT="/kaggle/working/new-model-2d-to-3d/image/stul.jpg" \
python -m openlrm.launch infer.lrm \
  --infer $INFER_CONFIG \
  model_name=$MODEL_NAME \
  image_input=$IMAGE_INPUT \
  export_video=$EXPORT_VIDEO \
  export_mesh=$EXPORT_MESH
```

## 5. Где искать результат

OpenLRM обычно сохраняет output внутри рабочей папки эксперимента/outputs. Найди так:

```python
!find /kaggle/working/OpenLRM -iname "*.obj" -o -iname "*.ply" -o -iname "*.glb" -o -iname "*.mp4" | head -50
```

## 6. Что написать в дипломе

Формулировка:

```text
В качестве сильной открытой reference-модели использована OpenLRM — открытая реализация Large Reconstruction Model.
OpenLRM использует pretrained DINOv2 encoder, transformer decoder для построения triplane-представления и neural field decoder для генерации 3D.
Модель имеет открытый код, pretrained weights и training pipeline, поэтому подходит для воспроизводимого сравнения.
```

## 7. Обучение OpenLRM

У OpenLRM есть training code:

```python
%cd /kaggle/working/OpenLRM

!accelerate launch \
  --config_file ./configs/accelerate-train.yaml \
  -m openlrm.launch train.lrm \
  --config ./configs/train-sample.yaml
```

Но честно:

```text
полноценное обучение OpenLRM требует подготовленного Objaverse/MVImgNet-like датасета и нескольких GPU.
На Kaggle это лучше использовать для inference/fine-tune experiments, а не для полного training from scratch.
```

## 8. Рекомендованная стратегия

```text
1. OpenLRM pretrained inference: красивый открытый baseline.
2. Твоя Hunyuan-like модель: собственная архитектура и обучение.
3. Сравнение:
   - наша модель: обучалась на ShapeNet Vox32, val_iou Stage1 около 0.80
   - OpenLRM: pretrained на Objaverse + MVImgNet, лучше generalization
```

Это будет выглядеть честно и убедительно.

## Источники

- OpenLRM GitHub: https://github.com/3DTopia/OpenLRM
- LRM paper: https://arxiv.org/abs/2311.04400
- OpenLRM model weights: https://huggingface.co/zxhezexin
