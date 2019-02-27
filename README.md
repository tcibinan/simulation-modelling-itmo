# Имитационное моделирование

Настоящий репозиторий содержит рабочие материалы по курсу *Имитационное моделирование*.

## Исследования

Выполнение программы исследований.

```bash
export PYTHONPATH=$PYTHONPATH:$PWD
python lab1/lab.py
```

## Отчеты

Сборка отчета по лабораторной работе осуществляется специфичным docker-образом, который работает в качестве
демона.

```bash
./start_daemoon.sh
./compile.sh lab1
./stop_daemon.sh
```

Сгенерированный отчет может быть найден как `outputs/lab1.pdf`.
