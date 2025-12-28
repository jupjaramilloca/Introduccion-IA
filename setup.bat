@echo off
echo Creando entorno virtual en Windows...
python -m venv venv

echo Activando entorno virtual...
call venv\Scripts\activate

echo Actualizando pip e instalando dependencias...
python -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo === Configuracion completada con exito ===
pause