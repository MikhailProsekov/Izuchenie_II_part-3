# АНАЛИЗ ДАННЫХ И ИСКУССТВЕННЫЙ ИНТЕЛЛЕКТ [in GameDev]
Отчет по лабораторной работе #3 выполнил:
- Просеков Михаил Павлович
- НМТ-210509
Отметка о выполнении заданий (заполняется студентом):

| Задание | Выполнение | Баллы |
| ------ | ------ | ------ |
| Задание 1 | # | 60 |
| Задание 2 | # | 20 |
| Задание 3 | # | 20 |

знак "*" - задание выполнено; знак "#" - задание не выполнено;

Работу проверили:
- к.т.н., доцент Денисов Д.В.
- к.э.н., доцент Панов М.А.
- ст. преп., Фадеев В.О.

[![N|Solid](https://cldup.com/dTxpPi9lDf.thumb.png)](https://nodesource.com/products/nsolid)

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

Структура отчета

- Данные о работе: название работы, фио, группа, выполненные задания.
- Цель работы.
- Задание 1.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Задание 2.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Задание 3.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Выводы.
- ✨Magic ✨

## Цель работы
Познакомиться с программными средствами для создания системы машинного обучения и ее интеграции в Unity.

## Задание 1
### Реализовать систему машинного обучения в связке Python - Google-Sheets – Unity. При выполнении задания можно использовать видео-материалы и исходные данные, предоставленные преподавателями курса.
Ход работы:
- Создайте новый пустой 3D проект на Unity.
- Скачайте папку с ML агентом. Вы найдете ее в облаке с исходными
файлами к лабораторной работе – ml-agents-release_19.
- В созданный проект добавьте ML Agent, выбрав Window - Package
Manager - Add Package from disk. Последовательно добавьте .json –
файлы:
- ml-agents-release_19 / com,unity.ml-agents / package.json
- ml-agents-release_19 / com,unity.ml-agents.extensions / package.json
- Если все сделано правильно, то во вкладке с компонентами
(Components) внутри Unity вы увидите строку ML Agent.
![image](https://user-images.githubusercontent.com/113620568/196039064-3ed1281b-efd6-496b-bb21-d359d947d3f6.png)

- Далее запускаем Anaconda Prompt для возможности запуска команд
через консоль.
- Далее пишем серию команд для создания и активации нового ML-
агента, а также для скачивания необходимых библиотек:
- mlagents 0.28.0;
- torch 1.7.1;
![image](https://user-images.githubusercontent.com/113620568/196039111-c78e5d01-d2e8-4861-886d-10ade84da3ac.png)
![image](https://user-images.githubusercontent.com/113620568/196039129-f17a5c9c-ca0c-49c8-b482-d7fbdd7d3d07.png).
```py
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class RollerAgent : Agent
{
    Rigidbody rBody;
    // Start is called before the first frame update
    void Start()
    {
        rBody = GetComponent<Rigidbody>();
    }

    public Transform Target;
    public override void OnEpisodeBegin()
    {
        if (this.transform.localPosition.y < 0)
        {
            this.rBody.angularVelocity = Vector3.zero;
            this.rBody.velocity = Vector3.zero;
            this.transform.localPosition = new Vector3(0, 0.5f, 0);
        }

        Target.localPosition = new Vector3(Random.value * 8-4, 0.5f, Random.value * 8-4);
    }
    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(Target.localPosition);
        sensor.AddObservation(this.transform.localPosition);
        sensor.AddObservation(rBody.velocity.x);
        sensor.AddObservation(rBody.velocity.z);
    }
    public float forceMultiplier = 10;
    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        Vector3 controlSignal = Vector3.zero;
        controlSignal.x = actionBuffers.ContinuousActions[0];
        controlSignal.z = actionBuffers.ContinuousActions[1];
        rBody.AddForce(controlSignal * forceMultiplier);

        float distanceToTarget = Vector3.Distance(this.transform.localPosition, Target.localPosition);

        if(distanceToTarget < 1.42f)
        {
            SetReward(1.0f);
            EndEpisode();
        }
        else if (this.transform.localPosition.y < 0)
        {
            EndEpisode();
        }
    }
}
```
- После этих действий запускаю MLAgent и обучаю его
![image](https://user-images.githubusercontent.com/113620568/196039577-3a79c626-9482-46e3-8863-62df17644947.png)
```py
behaviors:
  RollerBall:
    trainer_type: ppo
    hyperparameters:
      batch_size: 10
      buffer_size: 100
      learning_rate: 3.0e-4
      beta: 5.0e-4
      epsilon: 0.2
      lambd: 0.99
      num_epoch: 3
      learning_rate_schedule: linear
    network_settings:
      normalize: false
      hidden_units: 128
      num_layers: 2
    reward_signals:
      extrinsic:
        gamma: 0.99
        strength: 1.0
    max_steps: 500000
    time_horizon: 64
    summary_freq: 10000
```
В ходе выполнения программы куб хаотично респавнится в пределах плоскости,а шарик перемещается в сторону куба.



## Задание 2
### Подробно опишите каждую строку файла конфигурации нейронной сети

![image](https://user-images.githubusercontent.com/113620568/196040516-69c0a47c-f921-4ca3-a84e-0b94935a2eb2.png)

```py

behaviors:
  RollerBall:              #Для RollerBall
    trainer_type: ppo      #Обучение с подкреплением
    hyperparameters:        #Параметры обучения
      batch_size: 10        #Необходимо 10 циклов для обновления локальных экстремумов функции
      buffer_size: 100      #После 100 полных циклов агента создаем и обновляем (больше-лучше)
      learning_rate: 3.0e-4 #Шаг обучения(меньше-лучше)
      beta: 5.0e-4          #Случайность действий
      epsilon: 0.2          #Разность при обновлении локальных экстремумов старой и новой версии
      lambd: 0.99           #Авторитетность оценок значений во времени
      num_epoch: 3          #Количество повторений во время обновления локальных экстремумов
      learning_rate_schedule: linear  #График шага обучения
    network_settings:     #Сетевые настройки
      normalize: false  #Нормализация входных данных
      hidden_units: 128 #Скрытые нейроны
      num_layers: 2   #Скрытые слои для предыдущего параметра
    reward_signals:   #Вознаграждение
      extrinsic:
        gamma: 0.99  #Размер вознагреждения
        strength: 1.0   #Шаг для learning_rate_schedule
    max_steps: 500000  #Шаги моделирования в процессе обучения
    time_horizon: 64   #Количество циклов агента до ввода в модель
    summary_freq: 10000  #Суммарная частота

```

## Задание 3
### Какова роль параметра Lr? Ответьте на вопрос, приведите пример выполнения кода, который подтверждает ваш ответ. В качестве эксперимента можете изменить значение параметра.



## Выводы

В ходе выполнения данной лабораторной работы я изучил MLAgent, с помощью которого научил программу "преследовать" кубик. Выполнено это было с помощью машинного обучения. Сделала она это очень быстро, проигрывая одновременно несколько несколько сцен.

| Plugin | README |
| ------ | ------ |
| Dropbox | [plugins/dropbox/README.md][PlDb] |
| GitHub | [plugins/github/README.md][PlGh] |
| Google Drive | [plugins/googledrive/README.md][PlGd] |
| OneDrive | [plugins/onedrive/README.md][PlOd] |
| Medium | [plugins/medium/README.md][PlMe] |
| Google Analytics | [plugins/googleanalytics/README.md][PlGa] |

## Powered by

**BigDigital Team: Denisov | Fadeev | Panov**
