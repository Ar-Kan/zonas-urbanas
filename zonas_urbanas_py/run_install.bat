CALL conda activate ox

REM instalar biblioteca
REM pip install --user .

REM build inplace
python setup.py build_ext --inplace --build-lib "/build/"