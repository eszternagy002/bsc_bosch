set EPOCHS=15

echo Running training with -b0
python train.py -pt -e %EPOCHS% -b b0

echo Running training with -b1
python train.py -pt -e %EPOCHS% -b b1

echo Running training with -b2
python train.py -pt -e %EPOCHS% -b b2

echo Running training with -b3
python train.py -pt -e %EPOCHS% -b b3

echo Running training with -b4
python train.py -pt -e %EPOCHS% -b b4

echo Running training with -b5
python train.py -pt -e %EPOCHS% -b b5

echo Running training with -b6
python train.py -pt -e %EPOCHS% -b b6