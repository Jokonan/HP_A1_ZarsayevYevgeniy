




Результаты:

<img width="400" height="113" alt="image" src="https://github.com/user-attachments/assets/058ad3cd-118e-40ef-8905-1cfc1d565ac1" />



<img width="655" height="452" alt="image" src="https://github.com/user-attachments/assets/3bc8c50d-a36e-4da0-9168-c0cb31056e50" />


Блок схема:



Результаты:


<img width="342" height="122" alt="image" src="https://github.com/user-attachments/assets/e989c77d-5bac-44c8-9e4f-bd5035838d92" />




<img width="622" height="394" alt="image" src="https://github.com/user-attachments/assets/ac135293-b026-4b61-bdd1-f8379fedee2f" />


Блок схема:





# Контрольные вопросы


1.Какие основные типы памяти используются в OpenCL?

Ответ: Global - общая для всех рабочих потоков, медленная. Local - общая для группы потоков, быстрая. Private - индивидуальная для каждого потока, очень быстрая. Constant - только для чтения, для всех потоков.

2. Как настроить глобальную и локальную рабочую группу?

Ответ: Global - это общее количество операций, которые выполняют ядро. Local - это размер блока, который работает вместе и делит local память.

3. Чем отличается OpenCL от CUDA?

Ответ: CUDA работает только на NVIDIA GPU. OpenCL - кроссплатформенный, можно на CPU, GPU разных производителей.

4. Какие преимущества дает использование OpenCL?

Ответ: Можно запускать программы на разных устройствах. Параллельная обработка ускоряет вычисления. Можно использовать и CPU, и GPU одновременно.





### Сборка

!g++ task1.cpp -lOpenCL -O2 -o task1


### Запуск
!./task1

