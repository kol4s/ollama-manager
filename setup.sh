#!/usr/bin/env bash

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"

cd "$PROJECT_DIR"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creando entorno virtual..."
    python3 -m venv "$VENV_DIR"
fi

echo "Actualizando pip..."
"$VENV_DIR/bin/python" -m pip install --upgrade pip

echo "Instalando dependencias..."
"$VENV_DIR/bin/pip" install -r requirements.txt

echo "Verificando dependencias..."
"$VENV_DIR/bin/python" - <<'PY'
import PySide6
import psutil
import pynvml
import requests

print("Dependencias OK")
PY

chmod +x run.sh

echo
echo "Instalación completada."
echo "Arranque:"
echo "  ./run.sh"
