# Ollama Manager

Ollama Manager is a desktop application built with Python and PySide6 for managing, analyzing, optimizing, and monitoring local Ollama models.

> Unofficial community project. Not affiliated with or endorsed by Ollama.

## Features

- CPU, RAM and Swap detection
- NVIDIA GPU monitoring
- Intel graphics detection
- Multi-GPU awareness
- VRAM, temperature, utilization and power monitoring
- Ollama model management
- Pull, run, stop and delete models
- Hardware-aware parameter recommendations
- Editable proposed configurations
- Parameter validation
- Safe creation of optimized model variants
- Runtime monitoring
- Friendly connection, timeout and HTTP error handling

## Current Version

1.0.0

## Platform

The current version is primarily designed for Linux and Ollama.

## Requirements

- Linux
- Python 3
- Ollama installed and running
- Python virtual environment support

Python dependencies are listed in:

`requirements.txt`

## Installation

Clone the repository:

```bash
git clone https://github.com/kol4s/ollama-manager.git
cd ollama-manager
```

Run the setup script:

```bash
chmod +x setup.sh run.sh
./setup.sh
```

## Running Ollama Manager

```bash
./run.sh
```

## Running the Tests

```bash
.venv/bin/pytest -q
```

Current test status for version 1.0.0: **85 passed**

## Project Structure

```text
ollama-manager/
├── app.py
├── hardware_detector.py
├── core/
├── knowledge/
├── tests/
├── ui/
├── requirements.txt
├── pytest.ini
├── setup.sh
├── run.sh
├── VERSION
├── LICENSE
└── README.md
```

## Safety

Ollama Manager does not overwrite the original model when applying a recommended configuration.

Instead, it creates a new model variant with a unique name.

Example:

```text
gemma4:e4b
gemma4:e4b-optimized
gemma4:e4b-optimized-2
```

## Roadmap

Possible future development includes:

- Windows support
- llama.cpp backend
- GGUF model management
- automated benchmarks
- automatic performance tuning
- advanced multi-GPU optimization
- saved hardware profiles
- performance history
- model comparison
- additional inference backends

## License

This project is licensed under the MIT License.

See `LICENSE` for details.

## Disclaimer

Ollama Manager is an independent community project.

Ollama and related names belong to their respective owners.
