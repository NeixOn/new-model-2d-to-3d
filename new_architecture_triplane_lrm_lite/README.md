# Новая архитектура: Triplane Occupancy Reconstruction / LRM-lite

## Почему нужна новая архитектура

Текущая модель, которую мы обучали, является полезным baseline:

```text
RGB image -> 2D CNN encoder -> latent vector -> dense layer -> voxel grid 32x32x32
```

Она быстро обучается и дает понятную метрику `IoU`, но имеет принципиальные ограничения:

- выход ограничен грубой сеткой `32^3`;
- dense decoder не понимает локальную 3D-структуру;
- модель плохо восстанавливает тонкие детали;
- при долгом обучении быстро переобучается;
- качество на real images будет низким из-за synthetic-to-real gap.

Чтобы сделать архитектуру сильнее, лучше перейти от прямого предсказания `32^3` voxel-сетки к **неявному 3D-представлению**.

## Выбранный вариант

Рекомендуемая новая архитектура:

```text
Single RGB image
    -> pretrained image encoder
    -> triplane feature generator
    -> implicit occupancy/SDF decoder
    -> mesh extraction
```

Название папки:

```text
new_architecture_triplane_lrm_lite
```

Это компромисс между классическими implicit-моделями и современными LRM/TripoSR/CRM-подходами.

## Идея triplane

Вместо того чтобы хранить полный 3D feature volume, модель хранит три 2D feature planes:

```text
XY plane
XZ plane
YZ plane
```

Для любой 3D-точки:

```text
p = (x, y, z)
```

мы берём признаки из трёх плоскостей:

```text
f_xy = sample(XY, x, y)
f_xz = sample(XZ, x, z)
f_yz = sample(YZ, y, z)
```

Затем объединяем:

```text
f = concat(f_xy, f_xz, f_yz, positional_encoding(x, y, z))
```

И подаём в MLP:

```text
occupancy = MLP(f)
```

## Почему triplane лучше voxel baseline

### 1. Экономия памяти

Полный 3D volume:

```text
N x N x N
```

Triplane:

```text
3 x N x N
```

При `N=128`:

```text
3D volume: 128^3 = 2,097,152 cells
triplane: 3 * 128^2 = 49,152 cells
```

Это примерно в `42 раза` меньше пространственных ячеек.

### 2. Более высокое разрешение

Текущий baseline ограничен `32^3`. Triplane можно обучать с planes `64x64`, `128x128`, а mesh извлекать на сетке `64^3` или `128^3`.

### 3. Локальные признаки

Dense decoder из текущей модели сжимает всё изображение в один вектор. Triplane сохраняет пространственную структуру признаков, поэтому лучше восстанавливает:

- ножки стула;
- спинку;
- подлокотники;
- тонкие элементы;
- асимметрию.

### 4. Близость к современным моделям

LRM, TripoSR, CRM и InstantMesh используют похожие идеи:

- image encoder;
- latent 3D representation;
- triplane или sparse-view features;
- neural rendering / implicit decoding / mesh extraction.

Наша версия будет проще, но методологически ближе к современному уровню.

## Архитектура модели

### Вход

```text
image: RGB, 224x224 или 256x256
```

Для начала можно использовать `128x128`, но для лучшего качества лучше перейти на `224x224`.

### Image encoder

Варианты:

1. ResNet50 pretrained on ImageNet
2. EfficientNet-B0/B3
3. DINOv2 ViT-S/14 или ViT-B/14

Рекомендация:

```text
DINOv2 ViT-S/14 как encoder, если получится использовать pretrained weights.
```

Почему:

- сильные универсальные visual features;
- лучше переносится на real images;
- используется в современных reconstruction pipelines.

Если JAX/pretrained setup будет сложным, можно начать с:

```text
ResNet50 encoder
```

### Triplane generator

Encoder выдаёт image tokens/features.

Дальше возможны два варианта.

#### Вариант A: простой

```text
global image feature -> MLP -> triplane tensor
```

```text
triplane shape: 3 x C x H x W
```

Например:

```text
3 x 32 x 64 x 64
```

Плюсы:

- проще реализовать;
- быстрее обучить;
- хорош для первого эксперимента.

Минусы:

- меньше локальной привязки к изображению.

#### Вариант B: лучше

```text
image tokens -> transformer/cross-attention -> triplane tokens -> reshape to planes
```

Плюсы:

- ближе к LRM;
- лучше использует локальные части изображения;
- потенциально выше качество.

Минусы:

- сложнее код;
- больше памяти;
- дольше training.

Для следующего шага я рекомендую:

```text
Вариант A сначала, затем Вариант B.
```

### Implicit decoder

Вход:

```text
sampled triplane features + positional encoding(x, y, z)
```

Выход:

```text
occupancy probability
```

MLP:

```text
Linear -> GELU -> Linear -> GELU -> Linear -> sigmoid/logit
```

Лучше выдавать logits, а sigmoid применять только при метрике/inference.

### Mesh extraction

Для получения 3D-модели:

1. сэмплируем occupancy на сетке `64^3` или `128^3`;
2. применяем marching cubes;
3. сохраняем mesh в `.obj` или `.ply`.

## Loss functions

### Weighted BCE

Основная функция:

```text
weighted_binary_cross_entropy(pred_occupancy, gt_occupancy)
```

Почему weighted:

В 3D-пространстве пустых точек намного больше, чем занятых. Без веса модель может слишком часто предсказывать пустоту.

### Dice loss

Помогает оптимизировать overlap формы:

```text
Dice = 2 * intersection / (prediction + ground_truth)
```

### Eikonal loss, если перейти к SDF

Если вместо occupancy использовать SDF:

```text
||grad SDF|| = 1
```

Но для первого этапа лучше occupancy, потому что он ближе к текущей voxel-разметке.

## Метрики

Основные:

- IoU;
- Chamfer Distance;
- F-score;
- Normal Consistency.

Минимальный набор:

```text
IoU + Chamfer Distance
```

Для диплома лучше добавить:

```text
IoU, Chamfer-L1, F-Score@1%, F-Score@2%
```

## Данные

### Этап 1: ShapeNet / 3D-R2N2

Используем текущий датасет:

```text
ShapeNetRendering.tgz
ShapeNetVox32.tgz
```

Почему он подходит:

- уже подключён на Kaggle;
- есть RGB views;
- есть voxel ground truth;
- можно быстро сравнить с baseline;
- легко считать IoU.

Ограничение:

`ShapeNetVox32` даёт только `32^3` target. Для triplane occupancy это всё равно полезно на первом этапе, но качество будет ограничено target-разрешением.

### Этап 2: ShapeNet mesh -> own occupancy samples

Для более качественного обучения лучше использовать исходные mesh из ShapeNet, а не только `32^3` voxel.

Pipeline:

```text
mesh -> sample 3D points -> inside/outside labels -> train occupancy decoder
```

Это позволит обучаться не на грубой voxel-сетке, а на continuous occupancy.

### Этап 3: Objaverse

Objaverse интересен для более сильной модели.

Варианты:

1. Raw Objaverse  
   [Objaverse official](https://objaverse.com/datasets/objaverse.html)

2. Objaverse-HQ / G-buffer renderings  
   [Hugging Face: Objaverse-HQ-Rendering](https://huggingface.co/datasets/3DAIGC/Objaverse-HQ-Rendering)

3. Complete Objaverse multi-view renderings  
   [Hugging Face: complete-objaverse](https://huggingface.co/datasets/zeyuanyin/complete-objaverse)

4. Objaverse-Ortho10View  
   [Hugging Face: Objaverse-Ortho10View](https://huggingface.co/datasets/huanngzh/Objaverse-Ortho10View)

Для нас лучше не начинать с полного Objaverse-XL. Практичнее:

```text
отфильтрованный Objaverse subset + готовые render views + mesh assets
```

## Рекомендуемый план обучения

### Stage 1

Цель: доказать, что triplane occupancy работает лучше baseline.

Данные:

```text
ShapeNet chairs
MAX_MODELS_PER_CLASS=6778
VIEWS_PER_MODEL=4
```

Разрешение:

```text
image=128 или 224
occupancy queries из voxel32
```

Метрика:

```text
IoU на validation
```

Ожидаемый результат:

```text
val_iou > 0.45
```

### Stage 2

Цель: улучшить качество.

Изменения:

- больше views;
- image augmentations;
- stronger encoder;
- query sampling около поверхности;
- mesh extraction на `64^3`.

Ожидаемый результат:

```text
val_iou 0.50-0.60 на chairs
```

### Stage 3

Цель: generalization на разные классы.

Данные:

```text
chairs + cars + airplanes + tables + sofas
```

Метрика:

```text
mean IoU по категориям
Chamfer Distance
```

### Stage 4

Цель: переход к Objaverse.

Данные:

```text
Objaverse rendered subset
```

Нужны:

- rendered RGB images;
- camera metadata;
- mesh или occupancy/SDF labels;
- masks или alpha;
- фильтрация плохих объектов.

## Почему не сразу TripoSR/InstantMesh

Можно взять готовую TripoSR или InstantMesh, но это будет скорее использование pretrained модели, а не обучение собственной архитектуры.

Для диплома лучше:

1. показать baseline;
2. показать улучшенную собственную архитектуру;
3. сравнить с готовыми SOTA-моделями как external reference.

Triplane LRM-lite хорошо подходит как собственная архитектура:

- современнее voxel baseline;
- объяснима;
- реализуема на Kaggle;
- масштабируется на Objaverse;
- имеет понятные метрики.

## Итоговый выбор

Я рекомендую следующую архитектуру:

```text
Input RGB image
-> DINOv2/ResNet image encoder
-> triplane feature generator
-> implicit occupancy MLP
-> marching cubes
-> mesh OBJ/PLY
```

Это лучший следующий шаг после текущего baseline.

