# Отчет по текущей модели: JAX voxel baseline

## Краткое описание

Сегодня была обучена baseline-модель для задачи:

```text
одно RGB-изображение -> 3D voxel-реконструкция объекта
```

Текущий класс объектов:

```text
chair / стул
ShapeNet category id: 03001627
```

Основной рабочий файл:

```text
kaggle_tpu_v5e8_jax_quick_3d_recon.py
```

Inference-файл:

```text
kaggle_jax_infer_image_to_3d.py
```

Модель обучалась на Kaggle TPU v5e-8 через JAX:

```text
jax.pmap
8 TPU devices
```

## Почему использовали JAX

Были проверены разные варианты:

1. PyTorch/XLA
2. TensorFlow TPUStrategy
3. JAX pmap

PyTorch/XLA на Kaggle TPU v5e-8 падал с ошибкой:

```text
Expected 8 worker addresses, got 1
```

TensorFlow видел только CPU и падал при:

```text
ConfigureDistributedTPU
```

JAX корректно увидел все TPU-устройства:

```text
TPU_0
TPU_1
TPU_2
TPU_3
TPU_4
TPU_5
TPU_6
TPU_7
```

Поэтому для текущего baseline был выбран JAX.

## Датасет

Использованный датасет:

```text
ShapeNet 3D-R2N2
```

Файлы:

```text
ShapeNetRendering.tgz
ShapeNetVox32.tgz
```

Путь на Kaggle:

```text
/kaggle/input/datasets/sirish001/shapenet-3dr2n2
```

Что содержит датасет:

- RGB-рендеры объектов ShapeNet;
- voxel ground truth `32x32x32`;
- несколько видов на один 3D-объект.

На текущем этапе использовалась категория:

```text
chair: 03001627
```

Количество моделей в полной категории chairs:

```text
примерно 6778 3D-моделей
```

В ранних экспериментах использовались:

```text
600 моделей
1 изображение на модель
```

Затем был запущен более полный прогон:

```text
MAX_MODELS_PER_CLASS=6778
EPOCHS=200
PER_DEVICE_BATCH=4
LR=2e-4
```

На тот момент `VIEWS_PER_MODEL` ещё не был вынесен в окружение, поэтому фактически использовался:

```text
VIEWS_PER_MODEL=1
```

То есть модель обучалась примерно на:

```text
6778 изображений стульев
```

## Архитектура

Текущая модель намеренно простая.

Общий вид:

```text
RGB image 128x128x3
    -> Conv2D encoder
    -> global average pooling
    -> fully connected layers
    -> voxel logits 32x32x32
```

### Encoder

Encoder состоит из нескольких 2D convolution layers:

```text
Conv2D 3 -> 32
Conv2D 32 -> 64
Conv2D 64 -> 128
Conv2D 128 -> 256
Global Average Pooling
```

Каждый convolution block использует:

```text
ReLU activation
stride=2
```

Таким образом изображение сжимается в компактный feature-vector.

### Decoder

Decoder:

```text
Dense 256 -> 512
Dense 512 -> 32768
reshape -> 32x32x32x1
```

Выход модели:

```text
voxel logits
```

Важно: модель не выдаёт сразу `0` или `1`. Она выдаёт logits, которые потом проходят через sigmoid:

```text
probability = sigmoid(logit)
```

Затем применяется threshold:

```text
occupied = probability > 0.4
```

## Почему архитектура ограничена

Главная проблема:

```text
decoder не является настоящим 3D decoder
```

Он не строит форму постепенно через 3D convolution. Он просто предсказывает все `32768` voxel через dense layer.

Это значит:

- модель плохо знает локальную 3D-связность;
- ей трудно строить тонкие структуры;
- она склонна предсказывать усредненную форму;
- при долгом обучении она может запоминать train-объекты.

## Функция потерь

В текущем baseline использовалась комбинация:

```text
BCE loss + Dice loss
```

### BCE loss

Binary Cross Entropy применяется к каждому voxel:

```text
voxel occupied / voxel empty
```

Задача:

```text
научить модель правильно классифицировать каждый voxel
```

Если voxel должен быть занят, а модель предсказывает пустоту, loss растет. Если voxel должен быть пустой, а модель предсказывает объект, loss тоже растет.

### Dice loss

Dice loss помогает оптимизировать пересечение формы:

```text
Dice = 2 * intersection / (prediction + ground_truth)
```

И loss:

```text
Dice loss = 1 - Dice
```

Почему он нужен:

В voxel-сетке пустого пространства гораздо больше, чем занятых voxel. Если использовать только BCE, модель может стать слишком осторожной и предсказывать пустоту.

### Weighted BCE

В последней версии скрипта добавлен параметр:

```text
POS_WEIGHT
```

Он усиливает вклад занятых voxel.

Рекомендуемое значение:

```text
POS_WEIGHT=4.0
```

Смысл:

```text
ошибка на occupied voxel важнее, чем ошибка на empty voxel
```

Это должно помочь лучше восстанавливать тонкие части стульев:

- ножки;
- спинки;
- подлокотники.

## Метрики

### train_loss

Loss на обучающей выборке.

Показывает, насколько хорошо модель подгоняется под training data.

### val_loss

Loss на validation set.

Показывает, насколько хорошо модель работает на объектах, которых не видела во время обучения.

### train_iou

IoU на train set:

```text
intersection / union
```

Считается между:

```text
predicted voxel grid
ground truth voxel grid
```

### val_iou

Главная метрика.

Именно по ней выбиралась лучшая модель:

```text
best_model_params.npz
```

Если `val_iou` становился лучше, checkpoint сохранялся.

## Результаты экспериментов

### Эксперимент 1: 600 моделей, 100 эпох

Параметры:

```text
MAX_MODELS_PER_CLASS=600
VIEWS_PER_MODEL=1
EPOCHS=100
PER_DEVICE_BATCH=4
LR=2e-4
```

Результат:

```text
Best validation IoU: 0.3101
```

Вывод:

Модель училась, но маленький набор данных и простая архитектура ограничивали качество.

### Эксперимент 2: 6778 моделей, 200 эпох

Параметры:

```text
MAX_MODELS_PER_CLASS=6778
VIEWS_PER_MODEL=1
EPOCHS=200
PER_DEVICE_BATCH=4
LR=2e-4
```

Лучший результат:

```text
Best validation IoU: 0.4124
```

Лучшее значение было достигнуто примерно на:

```text
Epoch 89/200
```

На 200-й эпохе:

```text
train_iou = 0.6293
val_iou = 0.3765
```

Это означает переобучение.

## Что значит переобучение в нашем случае

Переобучение видно так:

```text
train_iou растёт
val_iou падает или не растёт
```

В нашем эксперименте:

```text
train_iou: 0.21 -> 0.63
val_iou: максимум 0.4124, потом падение до 0.3765
```

Это означает:

- модель всё лучше запоминает training chairs;
- но хуже обобщает на unseen validation chairs;
- архитектура недостаточно сильна для generalization;
- нужно больше разнообразных views или более правильное 3D-представление.

## Почему результат получился таким

### 1. Низкое voxel-разрешение

Target:

```text
32x32x32
```

Это всего:

```text
32768 voxel
```

Для стульев это мало. Ножки и тонкие детали занимают очень мало voxel и легко теряются.

### 2. Один вид на модель

В эксперименте использовалось:

```text
VIEWS_PER_MODEL=1
```

Но в датасете есть несколько rendered views. Один вид не раскрывает всю 3D-форму, особенно если часть объекта скрыта.

### 3. Простая архитектура

Модель не использует:

- ResNet encoder;
- pretrained features;
- 3D transposed convolutions;
- triplanes;
- implicit fields;
- local image features.

Она решает сложную задачу слишком простым способом.

### 4. Domain gap

Модель обучается на синтетических рендерах ShapeNet.

На реальных фотографиях:

- другой свет;
- другой фон;
- другой ракурс;
- шум;
- перспектива;
- occlusions.

Поэтому real-image качество будет ниже.

### 5. Невидимая геометрия

По одному изображению невозможно знать заднюю часть объекта точно. Модель должна угадывать её по статистике данных.

Это называется:

```text
shape prior
```

Если shape prior слабый, модель выдаёт усредненную форму.

## Как улучшить текущую модель

### Улучшение 1: больше views

Теперь параметр вынесен в env:

```text
VIEWS_PER_MODEL
```

Рекомендуемый запуск:

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

Ожидаемый эффект:

```text
лучше generalization
меньше переобучение
выше val_iou
```

### Улучшение 2: early stopping

В последней версии добавлен:

```text
PATIENCE
```

Если `val_iou` не улучшается долгое время, обучение остановится.

Это защищает от ситуации:

```text
best epoch = 89
training continues to 200
val quality degrades
```

### Улучшение 3: weighted BCE

Параметр:

```text
POS_WEIGHT=4.0
```

Должен улучшить восстановление занятых voxel.

### Улучшение 4: learning-rate decay

Добавлены параметры:

```text
LR
MIN_LR
LR_WARMUP_EPOCHS
```

Learning rate теперь может плавно уменьшаться, что помогает улучшить convergence.

### Улучшение 5: заменить decoder

Самое важное архитектурное улучшение:

```text
dense decoder -> 3D ConvTranspose decoder
```

Вместо:

```text
Dense -> 32768 voxel
```

лучше:

```text
latent -> 4x4x4xC -> Conv3DTranspose -> 8^3 -> 16^3 -> 32^3
```

Это позволит decoder учитывать локальную 3D-структуру.

### Улучшение 6: перейти к triplane/implicit field

Это рекомендуемая новая архитектура:

```text
image -> encoder -> triplane features -> implicit occupancy MLP -> mesh
```

Она описана в:

```text
new_architecture_triplane_lrm_lite/README.md
```

## Как использовать текущую модель

После обучения лучший checkpoint сохраняется:

```text
/kaggle/working/quick_3d_recon_results_jax/best_model_params.npz
```

Для проверки одной картинки:

```python
%cd /kaggle/working/new-model-2d-to-3d
%run kaggle_jax_infer_image_to_3d.py --image /path/to/image.png
```

Результаты:

```text
/kaggle/working/single_image_reconstruction/prediction.obj
/kaggle/working/single_image_reconstruction/prediction_preview.png
/kaggle/working/single_image_reconstruction/prediction_voxels.npy
```

Важно:

Текущая модель обучена на стульях. Лучше подавать:

- фото стула;
- объект по центру;
- простой фон;
- без лишних предметов;
- весь объект в кадре.

## Итог

Текущая модель достигла:

```text
Best validation IoU: 0.4124
```

Это нормальный результат для простого baseline, но недостаточный для качественной 3D reconstruction.

Главные причины ограничения:

- `32^3` voxel target;
- один rendered view на объект;
- dense decoder;
- отсутствие pretrained encoder;
- synthetic-only training data.

Следующий лучший шаг:

```text
Triplane Occupancy Reconstruction / LRM-lite
```

Он должен дать более высокое качество и лучше подходит для дальнейшего перехода на Objaverse.

