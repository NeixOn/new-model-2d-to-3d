# Быстрый запуск single-image 3D reconstruction на Kaggle TPU v5e-8

## 1. Датасет

В Kaggle Notebook нажми **Add input** и добавь датасет:

https://www.kaggle.com/datasets/sirish001/shapenet-3dr2n2

Внутри должны быть архивы:

- `/kaggle/input/shapenet-3dr2n2/ShapeNetRendering.tgz`
- `/kaggle/input/shapenet-3dr2n2/ShapeNetVox32.tgz`

Это удобный стартовый вариант, потому что там уже есть:

- RGB-рендеры объектов ShapeNet;
- готовые voxel-модели `32x32x32`;
- пары подходят для задачи "одно изображение -> 3D форма".

## 2. Настройки Kaggle

В правой панели ноутбука:

- Accelerator: `TPU v5e-8`
- Internet: можно `Off`, если датасет уже подключен через Add input
- Session options: стандартные

## 3. Запуск

Для Kaggle TPU v5e-8 сначала используй JAX-версию:

```python
%run kaggle_tpu_v5e8_jax_quick_3d_recon.py
```

Она использует `jax.pmap` и запускается параллельно на всех TPU-устройствах, которые видит JAX.

TensorFlow-версия тоже лежит в репозитории:

```python
%run kaggle_tpu_v5e8_tf_quick_3d_recon.py
```

Но в некоторых Kaggle v5e runtime TensorFlow видит только CPU и падает на `ConfigureDistributedTPU`.

PyTorch/XLA-версия тоже лежит в репозитории:

```python
%run kaggle_tpu_v5e8_quick_3d_recon.py
```

Но на текущем Kaggle v5e runtime она может падать на multi-process PJRT initialization с ошибкой `Expected 8 worker addresses, got 1`.

Загрузи файл в Kaggle Notebook или создай cell с содержимым файла.

Если файл лежит в `/kaggle/working`, запусти:

```python
%run /kaggle/working/new-model-2d-to-3d/kaggle_tpu_v5e8_jax_quick_3d_recon.py
```

Скрипт сам:

- распакует маленький subset из архивов;
- обучит модель `ResNet18 encoder -> 3D voxel decoder`;
- посчитает validation IoU;
- сохранит веса и preview-картинки.

Результаты будут здесь:

```text
/kaggle/working/quick_3d_recon_results/
```

## 4. Параметры для первого запуска

Параметры можно задавать прямо в Kaggle cell перед запуском:

```python
%env EPOCHS=100
%env MAX_MODELS_PER_CLASS=600
%env VIEWS_PER_MODEL=1
%env PER_DEVICE_BATCH=4
%env LR=2e-4
%run kaggle_tpu_v5e8_jax_quick_3d_recon.py
```

Или для более долгого запуска:

```python
%env EPOCHS=150
%env MAX_MODELS_PER_CLASS=1200
%env VIEWS_PER_MODEL=4
%env PER_DEVICE_BATCH=4
%env LR=2e-4
%run kaggle_tpu_v5e8_jax_quick_3d_recon.py
```

Значения по умолчанию внутри файла:

```python
MAX_MODELS_PER_CLASS = 600
VIEWS_PER_MODEL = 1
EPOCHS = 25
PER_DEVICE_BATCH = 4
LR = 2e-4
```

Для самого первого теста:

```python
%env MAX_MODELS_PER_CLASS=80
%env VIEWS_PER_MODEL=1
%env EPOCHS=2
%run kaggle_tpu_v5e8_jax_quick_3d_recon.py
```

Когда убедишься, что всё работает:

```python
%env MAX_MODELS_PER_CLASS=1200
%env VIEWS_PER_MODEL=4
%env EPOCHS=100
%run kaggle_tpu_v5e8_jax_quick_3d_recon.py
```

Для максимального прогона текущей архитектуры:

```python
%env EPOCHS=180
%env MAX_MODELS_PER_CLASS=6778
%env VIEWS_PER_MODEL=4
%env PER_DEVICE_BATCH=4
%env LR=2e-4
%env MIN_LR=2e-5
%env POS_WEIGHT=4.0
%env PATIENCE=35
%run kaggle_tpu_v5e8_jax_quick_3d_recon.py
```

И можно добавить классы:

```python
CLASSES = {
    "chair": "03001627",
    "airplane": "02691156",
    "car": "02958343",
}
```

## 5. Что смотреть после запуска

Главные файлы:

```text
/kaggle/working/quick_3d_recon_results/best_model.pt
/kaggle/working/quick_3d_recon_results/previews/sample_0_pred.png
/kaggle/working/quick_3d_recon_results/previews/sample_0_gt.png
```

`pred.png` и `gt.png` показывают три проекции voxel-сетки. Для диплома потом лучше добавить нормальный 3D-рендер через marching cubes, но для первого smoke test этих preview достаточно.

## 6. Проверка на одном изображении

После обучения можно подать путь к своему изображению:

```python
%cd /kaggle/working/new-model-2d-to-3d
%run kaggle_jax_infer_image_to_3d.py --image /kaggle/path/to/image.png
```

Результаты сохраняются сюда:

```text
/kaggle/working/single_image_reconstruction/
```

Главные файлы:

```text
prediction.obj
prediction_preview.png
prediction_voxels.npy
```

Если модель выдала слишком пустую форму, можно снизить threshold:

```python
%run kaggle_jax_infer_image_to_3d.py --image /kaggle/path/to/image.png --threshold 0.25
```
