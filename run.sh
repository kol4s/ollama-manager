#!/usr/bin/env bash

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
PYTHON="$VENV_DIR/bin/python"

cd "$PROJECT_DIR"

if [ ! -x "$PYTHON" ]; then
    echo "ERROR: no se encontró el entorno virtual en:"
    echo "  $VENV_DIR"
    echo
    echo "Crea el entorno e instala las dependencias antes de continuar."
    exit 1
fi

if ! "$PYTHON" -c \
    "import PySide6, psutil, pynvml, requests" \
    >/dev/null 2>&1
then
    echo "ERROR: faltan dependencias de la aplicación."
    echo
    echo "Ejecuta:"
    echo "  $VENV_DIR/bin/pip install -r requirements.txt"
    exit 1
fi

exec "$PYTHON" app.py
