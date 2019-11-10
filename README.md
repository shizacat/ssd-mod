# Описание

Реалиация SSD на исходниках от [NVIDIA](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Detection/SSD).

Цели:
- разобраться с алгоритмом SSD
- Возможность замены base network
- Возможность менеять входное разрешение (300x300, 512x512)
- Менять количество классов

В качестве framework используется pytorch.

# Использованы библиотеки

- https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools - для работы с coco. Устанавливать из git, а то в python3 работать не будет.