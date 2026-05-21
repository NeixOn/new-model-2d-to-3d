# Обзор моделей single-image 3D reconstruction

## Цель обзора

Этот документ описывает основные семейства моделей для задачи **трехмерной реконструкции объекта по одному RGB-изображению**. Обзор нужен не только как список архитектур, но и как объяснение, почему разные подходы дают разное качество, на каких данных они обучались, какие метрики используют и насколько быстро работают.

Важное ограничение: результаты из разных статей нельзя всегда сравнивать напрямую. Даже если везде написано `IoU`, модели могут использовать разные:

- разрешения voxel-сетки;
- категории ShapeNet;
- train/test split;
- thresholds для occupancy;
- тип выходного представления: voxel, point cloud, mesh, implicit field, NeRF, triplane, Gaussian splats;
- режим: single-view или multi-view.

Поэтому ниже я разделяю **сопоставимые ShapeNet benchmark-результаты** и **современные генеративные image-to-3D модели**, которые чаще оценивают через визуальное качество, novel-view synthesis и время генерации.

## Основные источники

Использованные источники:

1. 3D-R2N2: Choy et al., “3D-R2N2: A Unified Approach for Single and Multi-view 3D Object Reconstruction”, ECCV 2016.  
   [arXiv](https://arxiv.org/abs/1604.00449), [project/PDF](https://3d-r2n2.stanford.edu/main.pdf)

2. PSGN: Fan et al., “A Point Set Generation Network for 3D Object Reconstruction from a Single Image”, CVPR 2017.  
   [arXiv](https://arxiv.org/abs/1612.00603)

3. Pixel2Mesh: Wang et al., “Pixel2Mesh: Generating 3D Mesh Models from Single RGB Images”, ECCV 2018.  
   [CVF PDF](https://openaccess.thecvf.com/content_ECCV_2018/papers/Nanyang_Wang_Pixel2Mesh_Generating_3D_ECCV_2018_paper.pdf)

4. AtlasNet: Groueix et al., “AtlasNet: A Papier-Mache Approach to Learning 3D Surface Generation”, CVPR 2018.  
   [arXiv](https://arxiv.org/abs/1802.05384)

5. Occupancy Networks: Mescheder et al., “Occupancy Networks: Learning 3D Reconstruction in Function Space”, CVPR 2019.  
   [arXiv](https://arxiv.org/abs/1812.03828), [CVF PDF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Mescheder_Occupancy_Networks_Learning_3D_Reconstruction_in_Function_Space_CVPR_2019_paper.pdf)

6. Pix2Vox: Xie et al., “Pix2Vox: Context-Aware 3D Reconstruction From Single and Multi-View Images”, ICCV 2019.  
   [arXiv](https://arxiv.org/abs/1901.11153), [CVF PDF](https://openaccess.thecvf.com/content_ICCV_2019/papers/Xie_Pix2Vox_Context-Aware_3D_Reconstruction_From_Single_and_Multi-View_Images_ICCV_2019_paper.pdf)

7. Pix2Vox++: Xie et al., “Pix2Vox++: Multi-scale Context-aware 3D Object Reconstruction from Single and Multiple Images”, 2020.  
   [arXiv](https://arxiv.org/abs/2006.12250)

8. DISN: Xu et al., “DISN: Deep Implicit Surface Network for High-quality Single-view 3D Reconstruction”, NeurIPS 2019.  
   [NeurIPS page](https://papers.neurips.cc/paper/8340-disn-deep-implicit-surface-network-for-high-quality-single-view-3d-reconstruction), [arXiv](https://arxiv.org/abs/1905.10711)

9. LRM: Hong et al., “LRM: Large Reconstruction Model for Single Image to 3D”, ICLR 2024.  
   [project](https://yiconghong.me/LRM/), [arXiv](https://arxiv.org/abs/2311.04400)

10. TripoSR: “TripoSR: Fast 3D Object Reconstruction from a Single Image”, 2024.  
    [arXiv](https://arxiv.org/abs/2403.02151), [technical report PDF](https://stability.ai/s/TripoSR_report.pdf)

11. CRM: Wang et al., “CRM: Single Image to 3D Textured Mesh with Convolutional Reconstruction Model”, 2024.  
    [arXiv](https://arxiv.org/abs/2403.05034)

12. InstantMesh: Xu et al., “InstantMesh: Efficient 3D Mesh Generation from a Single Image with Sparse-view Large Reconstruction Models”, 2024.  
    [arXiv](https://arxiv.org/abs/2404.07191)

13. Objaverse and Objaverse-XL datasets.  
    [Objaverse](https://objaverse.com/datasets/objaverse.html), [Objaverse-XL arXiv](https://arxiv.org/abs/2307.05663)

14. G-buffer Objaverse / Objaverse-HQ renderings.  
    [Hugging Face dataset](https://huggingface.co/datasets/3DAIGC/Objaverse-HQ-Rendering)

15. Complete Objaverse multi-view rendering dataset.  
    [Hugging Face dataset](https://huggingface.co/datasets/zeyuanyin/complete-objaverse)

## Метрики

### IoU

`IoU` означает `Intersection over Union`. Для voxel-реконструкции:

```text
IoU = occupied(prediction) ∩ occupied(ground truth)
      /
      occupied(prediction) ∪ occupied(ground truth)
```

Плюсы:

- проста;
- хорошо подходит для `32^3` voxel baseline;
- часто используется в ShapeNet voxel papers.

Минусы:

- зависит от threshold;
- грубо оценивает поверхность;
- на низком разрешении `32^3` не видит тонкие детали;
- не всегда отражает визуальное качество mesh.

### Chamfer Distance

`Chamfer Distance` измеряет расстояние между двумя point clouds или sampled surface points. Чем меньше, тем лучше.

Плюсы:

- подходит для point cloud и mesh;
- чувствительнее к поверхности, чем voxel IoU.

Минусы:

- зависит от нормализации масштаба;
- разные статьи используют разные множители и sampling.

### F-Score

`F-Score@tau` считает, какая доля точек поверхности находится близко к ground truth при заданном пороге `tau`. Используется для mesh/point cloud.

Плюсы:

- хорошо отражает качество поверхности;
- понятнее для mesh, чем IoU.

Минусы:

- сильно зависит от выбранного порога.

### Normal Consistency

Сравнивает нормали предсказанной и правильной поверхности. Чем выше, тем лучше.

Плюсы:

- показывает качество ориентации поверхности;
- полезно для mesh.

Минусы:

- не всегда применимо к voxel без post-processing.

### Novel View Synthesis метрики

Современные LRM/TripoSR/CRM/InstantMesh часто оценивают не только 3D-геометрию, но и качество рендера новых видов:

- `PSNR`;
- `SSIM`;
- `LPIPS`;
- CLIP similarity;
- FID/KID в генеративных работах.

Это не то же самое, что точная геометрия. Модель может красиво рендерить, но иметь неточную mesh-топологию.

## Сравнительная таблица классических моделей

Здесь приведены ориентировочные результаты на ShapeNet. Значения взяты из оригинальных статей и сравнительных таблиц в работах Occupancy Networks, Pix2Vox/Pix2Vox++ и последующих multi-view работах. Я разделяю single-view и multi-view, потому что multi-view существенно проще.

| Модель | Год | Представление | Датасет | Метрики / качество | Скорость / ресурсы | Комментарий |
|---|---:|---|---|---|---|---|
| 3D-R2N2 | 2016 | voxel `32^3`, recurrent fusion | ShapeNet / 3D-R2N2, 13 категорий | single-view IoU около `0.49-0.56` в разных протоколах; multi-view до `~0.636` | обучение тяжелее современных baseline из-за 3D conv/recurrent, но inference быстрый | Сильный ранний baseline. Хорош для voxel, но ограничен низким разрешением и грубой геометрией. |
| PSGN | 2017 | point cloud | ShapeNet | Chamfer в сравнении ONet около `0.215` mean; IoU обычно не главная метрика | inference быстрый, нет 3D conv volume | Уходит от voxel, генерирует точки. Лучше поверхность, но нет явной mesh-топологии. |
| Pixel2Mesh | 2018 | deformable mesh | ShapeNet | mean IoU около `0.480`, Chamfer около `0.216`, Normal Consistency около `0.772` в сравнении ONet | медленнее point/voxel baseline из-за graph conv, но дает mesh напрямую | Хорош для mesh, но зависит от начальной сферы/эллипсоида и хуже работает при сложной топологии. |
| AtlasNet | 2018 | параметризованные surface patches | ShapeNet | Chamfer около `0.175`, Normal Consistency около `0.811` в сравнении ONet | inference умеренный, surface patches проще 3D volume | Лучше point cloud по поверхности, но patches могут давать разрывы/самопересечения. |
| Occupancy Networks | 2019 | implicit occupancy field | ShapeNet | mean IoU около `0.571`, Chamfer около `0.215`, Normal Consistency около `0.834` | inference медленнее voxel, потому что нужно query many 3D points + mesh extraction | Сильный скачок качества за счет continuous representation. Не ограничена `32^3`, но mesh extraction дорогой. |
| DISN | 2019 | implicit SDF + local image features | ShapeNet, Pix3D | в статье заявлены SOTA результаты на synthetic и real images; особенно сильна на деталях | inference дороже ONet из-за локальных features и SDF queries | Улучшает implicit field локальными 2D-признаками. Лучше восстанавливает тонкие структуры. |
| Pix2Vox | 2019 | voxel `32^3`, context-aware fusion | ShapeNet, Pix3D | single-view около `0.66` IoU; multi-view до `~0.706` | быстрее implicit mesh extraction, но voxel resolution ограничивает детали | Очень сильный voxel-подход. Context-aware fusion выбирает лучшие части реконструкции из разных видов. |
| Pix2Vox++ | 2020 | voxel `32^3`, multi-scale context | ShapeNet, Pix3D, Things3D | single-view около `0.670`, multi-view до `~0.719` IoU | эффективнее Pix2Vox при хорошем качестве | Улучшение Pix2Vox за счет multi-scale признаков и более эффективных модулей. |

## Почему качество отличается

### Voxel-модели

Примеры: 3D-R2N2, Pix2Vox, Pix2Vox++.

Сильные стороны:

- простая постановка задачи: каждый voxel занят/пуст;
- легко считать IoU;
- удобно обучать supervised на ShapeNet/3D-R2N2.

Слабые стороны:

- память растет кубически: `32^3`, `64^3`, `128^3`;
- тонкие детали теряются;
- mesh после marching cubes часто выглядит грубо.

Почему Pix2Vox лучше 3D-R2N2:

- более сильный 2D encoder;
- context-aware fusion;
- лучше использует информацию из изображений;
- специально оптимизирован под single/multi-view voxel reconstruction.

### Point cloud модели

Пример: PSGN.

Сильные стороны:

- не тратят память на пустое 3D-пространство;
- могут описывать поверхность точками;
- быстрый inference.

Слабые стороны:

- point cloud не является готовой 3D-моделью;
- нужен отдельный meshing;
- сложно гарантировать гладкую поверхность и топологию.

### Mesh-модели

Пример: Pixel2Mesh.

Сильные стороны:

- сразу дают mesh;
- удобно использовать в 3D редакторах;
- лучше подходят для поверхностей, чем voxel.

Слабые стороны:

- часто деформируют фиксированный template;
- сложная топология плохо восстанавливается;
- тонкие/раздельные части объекта могут быть проблемой.

### Implicit-модели

Примеры: Occupancy Networks, DISN.

Сильные стороны:

- continuous representation;
- можно получать mesh высокого разрешения;
- лучше тонкие структуры;
- не ограничены фиксированной voxel-сеткой.

Слабые стороны:

- inference требует много запросов к MLP;
- mesh extraction через marching cubes занимает время;
- обучение сложнее, нужно sampling 3D-точек.

Почему ONet/DISN часто лучше voxel baseline:

- они предсказывают не грубую `32^3` сетку, а функцию в 3D-пространстве;
- поверхность можно восстановить на более плотной сетке;
- DISN дополнительно использует local image features, что помогает деталям.

## Современные image-to-3D модели

Современные модели после 2023 года часто обучаются не только на ShapeNet, а на Objaverse/Objaverse-XL, MVImgNet, OmniObject3D, Google Scanned Objects и похожих наборах. Они решают более общую задачу: получить textured 3D asset из произвольного изображения.

| Модель | Год | Представление | Данные | Скорость | Качество / особенности |
|---|---:|---|---|---|---|
| Zero-1-to-3 | 2023 | novel-view diffusion | Objaverse renderings | медленнее feed-forward, потому что diffusion | Не сразу 3D mesh, а генерация новых видов. Часто используется как часть pipeline. |
| LRM | 2024 | transformer -> triplane NeRF | около 1 млн объектов: Objaverse + MVImgNet | около `5 секунд` на объект по project page | Большая модель, 500M параметров, хорошая generalization на in-the-wild images. |
| TripoSR | 2024 | transformer/triplane-like feed-forward | Objaverse + дополнительные данные | менее `0.5 секунды` по technical report | Очень быстрый mesh из одного изображения. Сильный practical baseline. |
| CRM | 2024 | generated orthographic views -> convolutional triplane -> FlexiCubes mesh | Objaverse-style training | около `10 секунд` по статье | Высокое качество textured mesh, лучше использует геометрические priors triplane. |
| InstantMesh | 2024 | sparse-view LRM pipeline | Objaverse / multi-view generated data | десятки секунд, зависит от multi-view generation | Сильная mesh generation архитектура, лучше визуально, но pipeline сложнее. |

## Почему современные модели выглядят лучше

Главная причина: они учатся на намного более разнообразных данных.

ShapeNet:

- аккуратные CAD-модели;
- ограниченные категории;
- синтетические рендеры;
- хорошо подходит для научного benchmark.

Objaverse:

- сотни тысяч объектов;
- разные стили, категории, материалы;
- ближе к реальным image-to-3D задачам;
- используется в LRM, TripoSR, CRM, Zero123-like работах.

Вторая причина: представление.

Современные модели обычно не предсказывают `32^3` voxel. Они используют:

- triplane features;
- neural radiance fields;
- signed distance / occupancy fields;
- FlexiCubes / differentiable meshing;
- multi-view diffusion as prior.

Это позволяет получить более гладкую форму и текстуру.

## Датасеты

### ShapeNet / 3D-R2N2

Что есть:

- CAD-модели;
- rendered RGB views;
- voxel targets `32^3`.

Плюсы:

- уже подходит для нашей задачи;
- легко считать IoU;
- удобно для дипломного baseline;
- быстрый старт на Kaggle.

Минусы:

- synthetic domain;
- мало разнообразия по сравнению с Objaverse;
- реальные фото с фоном работают хуже;
- voxel target ограничивает качество.

### Pix3D

Что есть:

- реальные изображения;
- aligned 3D CAD models;
- категории вроде chairs, sofas, tables.

Плюсы:

- можно проверять generalization на real images;
- важен для диплома как тест вне synthetic domain.

Минусы:

- меньше данных;
- сложнее подготовка;
- не лучший основной training dataset для большой модели.

### Objaverse

Что есть:

- более `800K` 3D objects;
- форматы GLTF/OBJ/FBX;
- разнообразные категории;
- лицензия CC-BY 4.0 для основной версии.

Плюсы:

- намного лучше для general image-to-3D;
- подходит для современных triplane/LRM-like моделей;
- можно рендерить свои views;
- есть готовые rendering datasets.

Минусы:

- качество объектов неоднородное;
- нужно фильтровать;
- raw Objaverse тяжелый;
- training pipeline сложнее ShapeNet.

### Objaverse-XL

Что есть:

- более `10M` 3D objects.

Плюсы:

- масштаб для foundation models;
- используется в современных Zero123/Objaverse-XL экспериментах.

Минусы:

- слишком большой для нашего текущего Kaggle-подхода;
- требуется агрессивная фильтрация;
- хранение и рендеринг могут стать главной проблемой.

### Готовые рендеры Objaverse

Полезные источники:

- [G-buffer Objaverse / Objaverse-HQ Rendering](https://huggingface.co/datasets/3DAIGC/Objaverse-HQ-Rendering)
- [Complete Objaverse multi-view rendering dataset](https://huggingface.co/datasets/zeyuanyin/complete-objaverse)
- [Objaverse-Ortho10View](https://huggingface.co/datasets/huanngzh/Objaverse-Ortho10View)

Эти наборы потенциально лучше raw Objaverse, потому что уже содержат rendered images. Но перед использованием нужно проверить:

- есть ли mesh/3D asset alongside rendered views;
- лицензии;
- размер;
- доступность на Kaggle;
- есть ли camera metadata;
- есть ли masks/depth/normal maps.

## Вывод по моделям

Для диплома разумно разделить путь на три уровня:

1. **Baseline:** наша текущая JAX voxel-модель.  
   Дает понятный supervised benchmark, быстро обучается, легко объясняется.

2. **Улучшенный supervised ShapeNet model:** triplane/implicit occupancy или Pix2Vox++-like voxel decoder.  
   Это лучший следующий шаг, если нужна собственная архитектура, а не просто запуск готового foundation model.

3. **Современный image-to-3D уровень:** LRM/TripoSR/CRM/InstantMesh style.  
   Это лучше визуально, но сложнее в обучении и требует Objaverse-like данных.

Моя рекомендация для следующей собственной архитектуры: **Triplane Occupancy Reconstruction, LRM-lite**.

Почему:

- значительно сильнее нашего dense voxel baseline;
- ближе к современным LRM/TripoSR/CRM;
- можно начать на ShapeNet/3D-R2N2;
- затем перенести на Objaverse-rendered data;
- можно получать mesh через marching cubes;
- не требует сразу обучать diffusion model.

