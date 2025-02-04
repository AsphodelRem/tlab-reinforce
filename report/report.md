# Отчет о проделанной работе

### Реализация метода REINFORCE \ baseline
В ходе работы был реализован метод REINFORCE с baseline в виде скользящего среднего. Проведены эксперименты по выравниванию модели с помощью двух разных моделей наград. Были сравнены результаты выровненной модели и SFT.


## Эксперимент 1
### Reward модель
- **Accuracy**: ~0.65

HF: https://huggingface.co/AsphodelRem/test-reward-model

Была обучена через обычный trl.RewardTrainer с добавлением margin.
Гиперпараметры находятся в файле reward_model.yaml

### Aligned sft
- **Средняя награда невыровненной модели**: ~0.15 
- **Средняя награда выровненной модели**: ~0.18

Все гиперпараметры указаны в конфиг-файлах.

Результат получился не самый хороших, хоть и видно тенденцию к росту средней награды. Я связываю это, скорее всего, с особенностями обучения с небольшим количеством видеопамяти: многие из промптов имеют большую длинну, сами ответы модели тоже выходят длинными, поэтому адекватно обучить на полной длине с паддингом не получилось, пришлось как обрезать промпты, так и сокращать количество генерируемых токенов. Это могло приводить к не совсем корректному виду подаваемых данных в reward модель, из-за чего фактическая полезность текста падала, соответственно, и награда тоже. Так же не получилось провести обучение достаточное количество времени, что так же сказалось на итоговых результатах. Но в ходе обучения выявлен тренд на повышение средней награды, так что можно считать, что процесс выравнивания, в целом, шёл.

## Эксперимент 2
### Реализация нового лосса

В качестве функции потерь был выбран и реализован pairwise loss.


##### Доп. идеи:
- Возможно, стоило покопать в сторону модели Plackett-Luce, но, к сожалению, не хватило времени на это.
- Как идея, можно было учесть "рейтинг" ответов как уверенность в правильной разметке ответа. То есть, если наша reward модель предсказывает 10 классов, то 0 класс - отклоненный ответ с самым высоким рейтингов, последний класс - положитльный ответ с самым высоким рейтингом. Все рейтинги в разметке просто приводим к общему диапазону (чтобы получить заданное количество классов). Соответственно, получим для каждого семпла разметку - класс, обозначающий "степень полезности ответа". 
Дальше можно идти разными путями, например, минимизировать KL между ответами и истинной разметкой (ну и все подобные способы решения таких задач)

### Reward модель
- **Pairwise accuracy**: ~0.63

HF: https://huggingface.co/AsphodelRem/test-custom-reward-model

### Aligned sft
- **Средняя награда невыровненной модели**: ~0.59
- **Средняя награда выровненной модели**: ~0.59 

Гиперпараметры (кроме количества классов и лосса для RM) не менялись

К сожалению, увеличить среднюю награду не получилось, причины, скорее всего, те же, что сверху + возможно, неправильная интеграция новой RM в метод.
Не совсем корректно в данной ситуации будет сравнивать выравнивание с двумя разными моделями наград, потому что фактическая награда вычисляется по-разному.


### Что получилось
1. Выявлен тренд на увеличение средней награды, что радует

### Что не получилось/проблемы:
1. Итоговое качество выравнивания, очевидно, хотелось бы получше. 
2. При реализации метода через наследование от Trainer сильно выросло потребление GPU памяти, в варианте реализации "с нуля" потребление было меньше.


### Bonus:
Отличия в реализации RLOO в trl:
Насколько я понял из кода и статьи, реализация RLOO в trl - это REINFORCE / A2C, где присутствует дополнтельно проход с помощью PPO.

