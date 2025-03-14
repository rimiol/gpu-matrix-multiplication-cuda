# Лабораторная работа №4: Использование графических процессоров для высокопроизводительных вычислений

## Цель работы
Изучение особенностей использования графических процессоров (GPU) для высокопроизводительных вычислений.

---

## Задание 1

### Описание задания
1. Разработать программу матричного умножения \( C = AB \) с использованием графического процессора и технологии CUDA. Тип элементов матрицы – `float` или `double`, в зависимости от возможностей графического процессора. Размеры матриц выбираются близкими к пределам объема оперативной памяти вычислительной системы.
2. Измерить время работы программы.
3. Предложить пути оптимизации программы для увеличения скорости вычислений.
4. Сравнить скорость вычислений оптимизированной и исходной программ. Объяснить наблюдаемые различия.
5. Составить отчет по результатам работы.

---

## Задание 2

### Описание задания
1. Модифицировать созданную в задании 1 программу для реализации алгоритма блочного матричного умножения.
2. Измерить время работы программы для различных размеров блоков. Предложить оптимальный с точки зрения производительности вариант программы. Объяснить причины повышения производительности в предложенном варианте.
3. Составить отчет по результатам работы.