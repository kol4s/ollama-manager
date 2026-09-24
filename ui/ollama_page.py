from PySide6.QtCore import Qt, QObject, QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QInputDialog,
    QListWidget,
    QListWidgetItem,
    QTabWidget,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QLineEdit,
)

from core.knowledge_base import KnowledgeBase
from core.ollama_client import (
    OllamaClient,
    OllamaConnectionError,
    OllamaTimeoutError,
    OllamaHTTPError,
)
from core.parameter_recommender import ParameterRecommender
from core.parameter_comparison import build_parameter_comparison
from core.parameter_diff import build_parameter_diff
from core.application_builder import build_application_plan_from_comparisons
from core.application_executor import apply_application_plan
from core.model_variant import (
    build_variant_name,
    build_available_variant_name,
)
from core.recommendation_engine import RecommendationEngine
from core.runtime_monitor import RuntimeMonitor
from core.parameter_validator import (
    ParameterValidationError,
    validate_parameters,
)


class PullWorker(QObject):
    progress = Signal(dict)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, client, model_name):
        super().__init__()
        self.client = client
        self.model_name = model_name

    def run(self):
        try:
            result = self.client.pull_model(
                self.model_name,
                progress_callback=self.progress.emit,
            )
            self.finished.emit(result)
        except Exception as exc:
            self.error.emit(
                str(exc)
            )


class ApplicationWorker(QObject):
    progress = Signal(dict)
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, client, plan):
        super().__init__()
        self.client = client
        self.plan = plan

    def run(self):
        try:
            result = apply_application_plan(
                self.client,
                self.plan,
                progress_callback=self.progress.emit,
            )
            self.finished.emit(result)
        except Exception as exc:
            self.error.emit(
                str(exc)
            )


class OllamaPage(QWidget):
    """Página de gestión y monitorización de Ollama."""

    def __init__(self):
        super().__init__()

        self.client = OllamaClient()
        self.runtime_monitor = RuntimeMonitor()

        self.layout = QVBoxLayout(self)

        title = QLabel("Ollama")
        title.setStyleSheet(
            "font-size: 24px; font-weight: bold;"
        )

        self.status_label = QLabel(
            "Estado: comprobando..."
        )
        self.status_label.setWordWrap(True)

        buttons_layout = QHBoxLayout()

        self.refresh_button = QPushButton(
            "Actualizar"
        )
        self.refresh_button.clicked.connect(
            self.refresh
        )

        self.analyze_button = QPushButton(
            "Analizar configuración"
        )
        self.analyze_button.clicked.connect(
            self._analyze_selected_model
        )

        self.run_button = QPushButton(
            "Ejecutar prueba"
        )
        self.run_button.clicked.connect(
            self._run_selected_model
        )

        self.delete_button = QPushButton(
            "Eliminar modelo"
        )
        self.delete_button.clicked.connect(
            self._delete_selected_model
        )

        self.pull_button = QPushButton(
            "Descargar modelo"
        )
        self.pull_button.clicked.connect(
            self._pull_model
        )

        self.stop_button = QPushButton(
            "Detener modelo"
        )
        self.stop_button.clicked.connect(
            self._stop_selected_model
        )

        self.pull_progress = QProgressBar()
        self.pull_progress.setRange(0, 100)
        self.pull_progress.setValue(0)
        self.pull_progress.setVisible(False)
        self.pull_progress.setFormat("%p%")

        self.apply_progress = QProgressBar()
        self.apply_progress.setRange(0, 100)
        self.apply_progress.setValue(0)
        self.apply_progress.setVisible(False)
        self.apply_progress.setFormat("%p%")

        buttons_layout.addWidget(
            self.refresh_button
        )
        buttons_layout.addWidget(
            self.analyze_button
        )
        buttons_layout.addWidget(
            self.run_button
        )
        buttons_layout.addWidget(
            self.delete_button
        )
        buttons_layout.addWidget(
            self.pull_button
        )
        buttons_layout.addWidget(
            self.stop_button
        )
        buttons_layout.addStretch()

        self.models_title = QLabel(
            "Modelos instalados"
        )
        self.models_title.setStyleSheet(
            "font-size: 18px; font-weight: bold;"
        )

        self.models_list = QListWidget()
        self.models_list.currentItemChanged.connect(
            self._model_selected
        )

        self.details = QTextEdit()
        self.details.setReadOnly(True)
        self.details.setPlaceholderText(
            "Selecciona un modelo para ver sus detalles."
        )

        self.summary_details = QTextEdit()
        self.summary_details.setReadOnly(True)

        self.hardware_details = QTextEdit()
        self.hardware_details.setReadOnly(True)

        self.recommendation_details = QTextEdit()
        self.recommendation_details.setReadOnly(True)

        self.parameter_details = QTextEdit()
        self.parameter_details.setReadOnly(True)

        self.apply_proposal_button = QPushButton(
            "Aplicar propuesta"
        )
        self.apply_proposal_button.clicked.connect(
            self._apply_configuration_proposal
        )

        self.review_changes_button = QPushButton(
            "Revisar cambios"
        )
        self.review_changes_button.clicked.connect(
            self._review_parameter_changes
        )

        self.parameter_table = QTableWidget()
        self.parameter_table.setColumnCount(5)
        self.parameter_table.setHorizontalHeaderLabels(
            [
                "Parámetro",
                "Actual",
                "Recomendado",
                "Propuesto",
                "Explicación",
            ]
        )
        self.parameter_table.setAlternatingRowColors(True)
        self.parameter_table.horizontalHeader().setSectionResizeMode(
            0,
            QHeaderView.ResizeToContents,
        )
        self.parameter_table.horizontalHeader().setSectionResizeMode(
            1,
            QHeaderView.ResizeToContents,
        )
        self.parameter_table.horizontalHeader().setSectionResizeMode(
            2,
            QHeaderView.ResizeToContents,
        )
        self.parameter_table.horizontalHeader().setSectionResizeMode(
            3,
            QHeaderView.ResizeToContents,
        )
        self.parameter_table.horizontalHeader().setSectionResizeMode(
            4,
            QHeaderView.Stretch,
        )
        self.parameter_table.setEditTriggers(
            QTableWidget.DoubleClicked
            | QTableWidget.EditKeyPressed
            | QTableWidget.SelectedClicked
        )

        self._parameter_comparisons = []

        self.context_label = QLabel(
            "Contexto para ejecución:"
        )

        self.context_input = QLineEdit("65000")

        self.prompt_label = QLabel("Prompt:")

        self.prompt_input = QLineEdit(
            "Responde únicamente: OK"
        )

        self.running_list = QListWidget()
        self.running_list.setMaximumHeight(140)

        self.runtime_details = QTextEdit()
        self.runtime_details.setReadOnly(True)
        self.runtime_details.setMinimumHeight(420)
        self.runtime_details.setPlaceholderText(
            "Monitorización de CPU, RAM, GPU y VRAM."
        )

        self.content_tabs = QTabWidget()

        # -------------------------------------------------
        # TAB MODELOS
        # -------------------------------------------------

        models_tab = QWidget()
        models_layout = QVBoxLayout(models_tab)

        models_layout.addWidget(
            self.models_title
        )
        models_layout.addWidget(
            self.models_list
        )

        self.content_tabs.addTab(
            models_tab,
            "Modelos",
        )

        # -------------------------------------------------
        # TAB ANALISIS
        # -------------------------------------------------

        analysis_tab = QWidget()
        analysis_layout = QVBoxLayout(
            analysis_tab
        )

        analysis_layout.addWidget(
            QLabel("Información y análisis")
        )

        analysis_layout.addWidget(
            self.context_label
        )
        analysis_layout.addWidget(
            self.context_input
        )

        analysis_layout.addWidget(
            self.prompt_label
        )
        analysis_layout.addWidget(
            self.prompt_input
        )

        self.analysis_tabs = QTabWidget()

        self.analysis_tabs.addTab(
            self.summary_details,
            "Resumen",
        )

        self.analysis_tabs.addTab(
            self.hardware_details,
            "Hardware",
        )

        self.analysis_tabs.addTab(
            self.recommendation_details,
            "Recomendación",
        )

        parameter_tab = QWidget()
        parameter_layout = QVBoxLayout(parameter_tab)

        parameter_layout.addWidget(
            self.apply_proposal_button
        )
        parameter_layout.addWidget(
            self.review_changes_button
        )
        parameter_layout.addWidget(
            self.parameter_table
        )

        self.analysis_tabs.addTab(
            parameter_tab,
            "Parámetros",
        )

        analysis_layout.addWidget(
            self.analysis_tabs
        )

        self.content_tabs.addTab(
            analysis_tab,
            "Análisis",
        )

        # -------------------------------------------------
        # TAB EJECUCION
        # -------------------------------------------------

        running_tab = QWidget()
        running_layout = QVBoxLayout(
            running_tab
        )

        running_title = QLabel(
            "Modelos actualmente en ejecución"
        )
        running_title.setStyleSheet(
            "font-size: 18px; font-weight: bold;"
        )

        running_layout.addWidget(
            running_title
        )
        running_layout.addWidget(
            self.running_list,
            0,
        )
        running_layout.addWidget(
            self.runtime_details,
            1,
        )

        self.content_tabs.addTab(
            running_tab,
            "Ejecución",
        )

        # -------------------------------------------------
        # LAYOUT PRINCIPAL
        # -------------------------------------------------

        self.layout.addWidget(title)
        self.layout.addWidget(
            self.status_label
        )
        self.layout.addLayout(
            buttons_layout
        )
        self.layout.addWidget(
            self.pull_progress
        )
        self.layout.addWidget(
            self.apply_progress
        )
        self.layout.addWidget(
            self.content_tabs
        )

        self.timer = QTimer(self)
        self.timer.setInterval(3000)
        self.timer.timeout.connect(
            self._refresh_running_models_only
        )
        self.timer.start()

        self._refresh_runtime_details()
        self.refresh()
    @staticmethod
    def _format_ollama_error(exc):
        if isinstance(exc, OllamaConnectionError):
            return (
                "No se pudo conectar con Ollama. "
                "Comprueba que el servicio está iniciado."
            )

        if isinstance(exc, OllamaTimeoutError):
            return (
                "Ollama está tardando demasiado en responder."
            )

        if isinstance(exc, OllamaHTTPError):
            return str(exc)

        return (
            f"{type(exc).__name__}: {exc}"
        )

    def refresh(self):
        self.refresh_button.setEnabled(False)

        try:
            installed = self.client.list_models()

            self.models_list.clear()

            for model in installed:
                size_gib = (
                    model.size_bytes
                    / (1024 ** 3)
                )

                item = QListWidgetItem(
                    f"{model.name} — "
                    f"{size_gib:.2f} GiB"
                )

                item.setData(32, model.name)
                self.models_list.addItem(item)

            self._refresh_running_models_only()

            if installed:
                self.models_list.setCurrentRow(0)
            else:
                self.details.clear()

        except Exception as exc:
            self.status_label.setText(
                "Estado: Ollama no disponible\n"
                + self._format_ollama_error(exc)
            )

            self.models_list.clear()
            self.running_list.clear()
            self.details.clear()

            self.models_list.addItem(
                "No se pudieron consultar los modelos."
            )

        finally:
            self.refresh_button.setEnabled(True)

    def _model_selected(
        self,
        current: QListWidgetItem | None,
        previous: QListWidgetItem | None,
    ):
        del previous

        if current is None:
            self.details.clear()
            return

        model_name = current.data(32)

        if not model_name:
            self.details.clear()
            return

        try:
            model = self.client.show_model(
                str(model_name)
            )

            details = model.details

            parameter_size = details.get(
                "parameter_size",
                "N/D",
            )

            quantization = details.get(
                "quantization_level",
                "N/D",
            )

            family = details.get(
                "family",
                "N/D",
            )

            text = (
                f"Nombre: {model.name}\n"
                f"Familia: {family}\n"
                f"Arquitectura: "
                f"{model.architecture or 'N/D'}\n"
                f"Parámetros: {parameter_size}\n"
                f"Cuantización: {quantization}\n"
                f"Contexto: "
                f"{model.context_length:,}\n"
                f"Tamaño: "
                f"{model.size_bytes / (1024 ** 3):.2f} GiB\n"
                f"Parameter count: "
                f"{model.parameter_count:,}\n"
                f"Embedding length: "
                f"{model.embedding_length}\n"
                f"Block count: "
                f"{model.block_count}\n"
                f"Attention heads: "
                f"{model.head_count}\n"
                f"KV heads: "
                f"{model.head_count_kv}\n"
            )

            self.details.setPlainText(text)

            # Analizar automáticamente la configuración
            # actual usando el contexto seleccionado.
            self._analyze_selected_model(
                show_error=False
            )

        except Exception as exc:
            self.details.setPlainText(
                "No se pudo obtener la información "
                "detallada.\n"
                f"{type(exc).__name__}: {exc}"
            )

    def _refresh_runtime_details(self):
        try:
            snapshot = self.runtime_monitor.snapshot()
        except Exception as exc:
            self.runtime_details.setPlainText(
                "No se pudo obtener la monitorización."
                + chr(10)
                + f"{type(exc).__name__}: {exc}"
            )
            return

        lines = [
            "MONITORIZACIÓN DEL SISTEMA",
            "==========================",
            f"CPU: {snapshot.cpu_percent:.1f} %",
            f"RAM: {snapshot.ram_used_percent:.1f} %",
            (
                f"RAM usada: "
                f"{snapshot.ram_used_bytes / (1024 ** 3):.2f} GiB / "
                f"{snapshot.ram_total_bytes / (1024 ** 3):.2f} GiB"
            ),
            (
                f"RAM disponible: "
                f"{snapshot.ram_available_bytes / (1024 ** 3):.2f} GiB"
            ),
            (
                f"Swap: {snapshot.swap_used_percent:.1f} % "
                f"({snapshot.swap_used_bytes / (1024 ** 3):.2f} GiB / "
                f"{snapshot.swap_total_bytes / (1024 ** 3):.2f} GiB)"
            ),
        ]

        if not snapshot.gpus:
            lines.extend(
                [
                    "",
                    "GPU: no detectada",
                ]
            )
        else:
            for index, gpu in enumerate(snapshot.gpus, 1):
                temperature = (
                    f"{gpu.temperature_c:.1f} °C"
                    if gpu.temperature_c is not None
                    else "N/D"
                )
                utilization = (
                    f"{gpu.utilization_percent:.1f} %"
                    if gpu.utilization_percent is not None
                    else "N/D"
                )
                power = (
                    f"{gpu.power_watts:.1f} W"
                    if gpu.power_watts is not None
                    else "N/D"
                )

                if gpu.vram_total_bytes:
                    vram = (
                        f"{gpu.vram_used_percent:.1f} % "
                        f"({gpu.vram_used_bytes / (1024 ** 3):.2f} GiB / "
                        f"{gpu.vram_total_bytes / (1024 ** 3):.2f} GiB)"
                    )
                else:
                    vram = "N/D"

                lines.extend(
                    [
                        "",
                        f"GPU {index}: {gpu.model}",
                        f"  Fabricante: {gpu.vendor}",
                        f"  VRAM: {vram}",
                        f"  Temperatura: {temperature}",
                        f"  Utilización: {utilization}",
                        f"  Potencia: {power}",
                    ]
                )

        self.runtime_details.setPlainText(
            chr(10).join(lines)
        )

    def _refresh_running_models_only(self):
        self._refresh_runtime_details()
        self.running_list.clear()

        try:
            running_models = (
                self.client.list_running_models()
            )
        except Exception as exc:
            self.running_list.addItem(
                "Error consultando modelos activos: "
                f"{exc}"
            )
            return

        if not running_models:
            self.running_list.addItem(
                "No hay modelos actualmente "
                "en ejecución."
            )

            self.status_label.setText(
                "Estado: Ollama conectado · "
                "0 modelos activos"
            )
            return

        knowledge_base = KnowledgeBase(
            "knowledge/hardware_profiles.json"
        )
        knowledge_base.load()

        engine = RecommendationEngine(
            knowledge_base
        )

        for model in running_models:
            recommendation = (
                engine.analyze_running_model(model)
            )

            size_gib = (
                model.size_bytes
                / (1024 ** 3)
            )

            vram_gib = (
                model.size_vram_bytes
                / (1024 ** 3)
            )

            cpu_gib = (
                model.cpu_memory_bytes
                / (1024 ** 3)
            )

            text = (
                f"{model.name}\n"
                f"Contexto: "
                f"{model.context_length:,}\n"
                f"Tamaño total: "
                f"{size_gib:.2f} GiB\n"
                f"VRAM: "
                f"{vram_gib:.2f} GiB "
                f"({model.vram_percentage:.2f}%)\n"
                f"CPU/RAM: "
                f"{cpu_gib:.2f} GiB\n"
                f"Estrategia: "
                f"{recommendation.data['strategy']}\n"
                f"Evaluación: "
                f"{recommendation.title}"
            )

            item = QListWidgetItem(text)
            item.setData(32, model.name)
            self.running_list.addItem(item)

        self.status_label.setText(
            "Estado: Ollama conectado · "
            f"{len(running_models)} modelo(s) activo(s)"
        )

    def _selected_model_name(self) -> str | None:
        item = self.models_list.currentItem()

        if item is None:
            return None

        value = item.data(32)

        if not value:
            return None

        return str(value)

    def _analyze_selected_model(
        self,
        show_error: bool = True,
    ):
        model_name = self._selected_model_name()

        if not model_name:
            if show_error:
                QMessageBox.warning(
                    self,
                    "Modelo",
                    "Selecciona primero un modelo.",
                )
            return

        try:
            context = int(
                self.context_input.text().strip()
            )
        except ValueError:
            if show_error:
                QMessageBox.warning(
                    self,
                    "Contexto",
                    "El contexto debe ser un número entero.",
                )
            return

        if context <= 0:
            if show_error:
                QMessageBox.warning(
                    self,
                    "Contexto",
                    "El contexto debe ser mayor que cero.",
                )
            return

        try:
            ollama_model = self.client.show_model(
                model_name
            )

            model_info = self.client.to_model_info(
                ollama_model
            )

            from hardware_detector import HardwareDetector

            hardware = (
                HardwareDetector().detect_normalized()
            )

            knowledge_base = KnowledgeBase(
                "knowledge/hardware_profiles.json"
            )
            knowledge_base.load()

            engine = RecommendationEngine(
                knowledge_base
            )

            recommendations = engine.analyze_model(
                hardware,
                model_info,
                context_length=context,
            )

            recommender = ParameterRecommender(
                knowledge_base
            )

            profile = recommender.recommend(
                hardware,
                model_info,
                context_length=context,
            )

            self._parameter_comparisons = (
                build_parameter_comparison(
                    ollama_model.current_parameters,
                    profile.parameters,
                )
            )

            # ---------------------------------------------
            # RESUMEN
            # ---------------------------------------------

            summary = [
                "RESUMEN",
                "========================================",
                f"Modelo: {model_name}",
                f"Contexto solicitado: {context:,}",
                (
                    "Arquitectura: "
                    f"{ollama_model.architecture or 'N/D'}"
                ),
                (
                    "Parámetros: "
                    f"{ollama_model.details.get('parameter_size', 'N/D')}"
                ),
                (
                    "Cuantización: "
                    f"{ollama_model.details.get('quantization_level', 'N/D')}"
                ),
                (
                    "Tamaño: "
                    f"{ollama_model.size_bytes / (1024 ** 3):.2f} GiB"
                ),
                "",
            ]

            for recommendation in recommendations:
                summary.append(
                    f"[{recommendation.severity.upper()}] "
                    f"{recommendation.title}"
                )
                summary.append(
                    recommendation.explanation
                )

            # ---------------------------------------------
            # HARDWARE
            # ---------------------------------------------

            hardware_lines = [
                "HARDWARE DETECTADO",
                "========================================",
                f"CPU: {hardware.cpu.model}",
                (
                    "Núcleos físicos: "
                    f"{hardware.cpu.physical_cores}"
                ),
                (
                    "Hilos: "
                    f"{hardware.cpu.logical_cores}"
                ),
                (
                    "RAM total: "
                    f"{hardware.memory.total_bytes / (1024 ** 3):.2f} GiB"
                ),
                (
                    "RAM disponible: "
                    f"{hardware.memory.available_bytes / (1024 ** 3):.2f} GiB"
                ),
                (
                    "Swap total: "
                    f"{hardware.memory.swap_total_bytes / (1024 ** 3):.2f} GiB"
                ),
                (
                    "Swap usada: "
                    f"{hardware.memory.swap_used_bytes / (1024 ** 3):.2f} GiB"
                ),
                "",
            ]

            for index, gpu in enumerate(
                hardware.gpus,
                start=1,
            ):
                hardware_lines.extend(
                    [
                        f"GPU {index}: {gpu.model}",
                        f"  Vendor: {gpu.vendor}",
                        (
                            "  VRAM total: "
                            f"{gpu.vram_total_bytes / (1024 ** 3):.2f} GiB"
                        ),
                        (
                            "  VRAM usada: "
                            f"{gpu.vram_used_bytes / (1024 ** 3):.2f} GiB"
                        ),
                        (
                            "  VRAM libre: "
                            f"{gpu.vram_free_bytes / (1024 ** 3):.2f} GiB"
                        ),
                        (
                            "  Dedicada: "
                            f"{'Sí' if gpu.dedicated else 'No'}"
                        ),
                        (
                            "  API: "
                            f"{', '.join(gpu.compute_api) or 'N/D'}"
                        ),
                        f"  Driver: {gpu.driver or 'N/D'}",
                        "",
                    ]
                )

            # ---------------------------------------------
            # RECOMENDACION
            # ---------------------------------------------

            recommendation_lines = [
                "RECOMENDACIÓN",
                "========================================",
                "",
            ]

            for recommendation in recommendations:
                recommendation_lines.append(
                    f"Nivel: {recommendation.severity.upper()}"
                )
                recommendation_lines.append(
                    f"Conclusión: {recommendation.title}"
                )
                recommendation_lines.append(
                    recommendation.explanation
                )

                for key, value in recommendation.data.items():
                    if key == "estimated_memory_bytes":
                        recommendation_lines.append(
                            "Memoria estimada: "
                            f"{value / (1024 ** 3):.2f} GiB"
                        )
                    elif key == "vram_bytes":
                        recommendation_lines.append(
                            "VRAM total de la GPU: "
                            f"{value / (1024 ** 3):.2f} GiB"
                        )
                    elif key == "vram_used_bytes":
                        recommendation_lines.append(
                            "VRAM usada actualmente: "
                            f"{value / (1024 ** 3):.2f} GiB"
                        )
                    elif key == "vram_free_bytes":
                        recommendation_lines.append(
                            "VRAM libre actualmente: "
                            f"{value / (1024 ** 3):.2f} GiB"
                        )
                    elif key == "available_vram_bytes":
                        recommendation_lines.append(
                            "VRAM disponible para el modelo: "
                            f"{value / (1024 ** 3):.2f} GiB"
                        )
                    else:
                        recommendation_lines.append(
                            f"{key}: {value}"
                        )

                recommendation_lines.append("")

            # ---------------------------------------------
            # PARAMETROS
            # ---------------------------------------------

            parameter_lines = [
                "PARÁMETROS RECOMENDADOS",
                "========================================",
                "",
                f"Estado de capacidad: {profile.fit_status}",
                (
                    "VRAM disponible para la recomendación: "
                    f"{profile.available_vram_bytes / (1024 ** 3):.2f} GiB"
                ),
                (
                    "Memoria estimada del modelo: "
                    f"{profile.estimated_memory_bytes / (1024 ** 3):.2f} GiB"
                ),
                (
                    "Estrategia de ejecución: "
                    f"{profile.execution_strategy}"
                ),
                "",
            ]

            for parameter in profile.parameters:
                parameter_lines.append(
                    f"{parameter.name} = {parameter.value}"
                )
                parameter_lines.append(
                    f"  {parameter.explanation}"
                )
                parameter_lines.append("")

            self.parameter_table.setRowCount(
                len(self._parameter_comparisons)
            )

            for row, comparison in enumerate(
                self._parameter_comparisons
            ):
                values = [
                    comparison.name,
                    (
                        "N/D"
                        if comparison.current is None
                        else str(comparison.current)
                    ),
                    str(comparison.recommended),
                    str(comparison.proposed),
                    comparison.explanation,
                ]

                for column, value in enumerate(values):
                    item = QTableWidgetItem(value)

                    if column != 3:
                        item.setFlags(
                            item.flags()
                            & ~Qt.ItemIsEditable
                        )

                    self.parameter_table.setItem(
                        row,
                        column,
                        item,
                    )

            # ---------------------------------------------
            # ACTUALIZAR GUI
            # ---------------------------------------------

            self.details.setPlainText(
                "\n".join(summary)
            )

            self.summary_details.setPlainText(
                "\n".join(summary)
            )

            self.hardware_details.setPlainText(
                "\n".join(hardware_lines)
            )

            self.recommendation_details.setPlainText(
                "\n".join(recommendation_lines)
            )

            self.parameter_details.setPlainText(
                "\n".join(parameter_lines)
            )


        except Exception as exc:
            if show_error:
                QMessageBox.critical(
                    self,
                    "Error de análisis",
                    (
                        "No se pudo generar el análisis.\n"
                        f"{type(exc).__name__}: {exc}"
                    ),
                )

    def _apply_configuration_proposal(self):
        if not self._parameter_comparisons:
            QMessageBox.information(
                self,
                "Aplicar propuesta",
                "Analiza primero un modelo.",
            )
            return

        model_name = self._selected_model_name()

        if not model_name:
            QMessageBox.warning(
                self,
                "Modelo",
                "Selecciona primero un modelo.",
            )
            return

        try:
            model = self.client.show_model(
                model_name
            )

            plan = build_application_plan_from_comparisons(
                model,
                self._parameter_comparisons,
            )

            knowledge_base = KnowledgeBase(
                "knowledge/hardware_profiles.json"
            )
            knowledge_base.load()

            try:
                validate_parameters(
                    plan.parameters,
                    knowledge_base.data,
                )
            except ParameterValidationError as exc:
                QMessageBox.warning(
                    self,
                    "Parámetro no válido",
                    str(exc),
                )
                return

            target_exists = self.client.model_exists(
                plan.target_model
            )

            if target_exists:
                existing_names = {
                    item.name
                    for item in self.client.list_models()
                }

                plan.target_model = build_available_variant_name(
                    model,
                    existing_names=existing_names,
                )

            answer = QMessageBox.question(
                self,
                "Confirmar aplicación",
                (
                    f"Se creará una nueva variante:\n\n"
                    f"Original: {plan.source_model}\n"
                    f"Destino: {plan.target_model}\n\n"
                    "El modelo original NO será sobrescrito.\n\n"
                    "¿Deseas continuar?"
                ),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )

            if answer != QMessageBox.Yes:
                return

            self.apply_proposal_button.setEnabled(False)
            self.apply_progress.setVisible(True)
            self.apply_progress.setValue(0)

            self.status_label.setText(
                f"Estado: creando '{plan.target_model}'..."
            )

            self.application_thread = QThread(self)
            self.application_worker = ApplicationWorker(
                self.client,
                plan,
            )

            self.application_worker.moveToThread(
                self.application_thread
            )

            self.application_thread.started.connect(
                self.application_worker.run
            )

            self.application_worker.progress.connect(
                self._application_progress
            )

            self.application_worker.finished.connect(
                self._application_finished
            )

            self.application_worker.error.connect(
                self._application_error
            )

            self.application_worker.finished.connect(
                self.application_thread.quit
            )

            self.application_worker.error.connect(
                self.application_thread.quit
            )

            self.application_thread.finished.connect(
                self.application_worker.deleteLater
            )

            self.application_thread.finished.connect(
                self.application_thread.deleteLater
            )

            self.application_thread.start()

        except Exception as exc:
            if isinstance(exc, ValueError):
                detail = str(exc)
            else:
                detail = (
                    f"{type(exc).__name__}: {exc}"
                )

            QMessageBox.critical(
                self,
                "Error",
                (
                    "No se pudo preparar la aplicación.\n"
                    f"{detail}"
                ),
            )

    def _application_progress(self, data):
        status = str(
            data.get("status", "")
        )

        completed = int(
            data.get("completed", 0) or 0
        )

        total = int(
            data.get("total", 0) or 0
        )

        if total > 0:
            percent = int(
                completed * 100 / total
            )
            self.apply_progress.setValue(
                max(0, min(percent, 100))
            )

        if status:
            self.status_label.setText(
                f"Estado: {status}"
            )

    def _application_finished(self, result):
        self.apply_progress.setValue(100)
        self.apply_progress.setVisible(False)
        self.apply_proposal_button.setEnabled(True)

        if not result.success:
            QMessageBox.warning(
                self,
                "Aplicación",
                (
                    "Ollama terminó la operación sin indicar "
                    "éxito.\n\n"
                    f"Estado: {result.status}"
                ),
            )
            return

        # Verificación explícita de que la variante existe.
        try:
            installed = self.client.list_models()
            names = {
                model.name
                for model in installed
            }

            exists = (
                result.target_model
                in names
            )

        except Exception as exc:
            QMessageBox.critical(
                self,
                "Verificación",
                (
                    "La variante fue creada, pero no se pudo "
                    "verificar en la lista de modelos.\n"
                    f"{type(exc).__name__}: {exc}"
                ),
            )
            return

        if exists:
            self.status_label.setText(
                "Estado: variante creada correctamente"
            )

            QMessageBox.information(
                self,
                "Aplicación completada",
                (
                    f"Se ha creado la variante:\n\n"
                    f"{result.target_model}\n\n"
                    "El modelo original permanece intacto."
                ),
            )

            self.refresh()
        else:
            QMessageBox.warning(
                self,
                "Verificación",
                (
                    "Ollama informó de una finalización correcta, "
                    "pero la nueva variante no aparece en la lista."
                ),
            )

    def _application_error(self, message):
        self.apply_progress.setVisible(False)
        self.apply_proposal_button.setEnabled(True)

        self.status_label.setText(
            "Estado: error creando la variante"
        )

        QMessageBox.critical(
            self,
            "Error de aplicación",
            (
                "No se pudo crear la variante.\n"
                f"{message}"
            ),
        )

    def _review_parameter_changes(self):
        if not self._parameter_comparisons:
            QMessageBox.information(
                self,
                "Revisión de cambios",
                "No hay una configuración analizada.",
            )
            return

        for row, comparison in enumerate(
            self._parameter_comparisons
        ):
            item = self.parameter_table.item(
                row,
                3,
            )

            if item is not None:
                comparison.proposed = item.text()

        changes = build_parameter_diff(
            self._parameter_comparisons
        )

        if not changes:
            QMessageBox.information(
                self,
                "Revisión de cambios",
                "No hay cambios respecto a la configuración actual.",
            )
            return

        lines = [
            "CAMBIOS PROPUESTOS",
            "========================================",
            "",
        ]

        for change in changes:
            lines.append(
                f"{change.name}: "
                f"{change.current} → {change.proposed}"
            )

        lines.extend(
            [
                "",
                "Estos cambios todavía NO se han aplicado.",
            ]
        )

        QMessageBox.information(
            self,
            "Revisión de cambios",
            "\n".join(lines),
        )

    def _run_selected_model(self):
        model_name = self._selected_model_name()

        if not model_name:
            QMessageBox.warning(
                self,
                "Modelo",
                "Selecciona primero un modelo.",
            )
            return

        try:
            context = int(
                self.context_input.text().strip()
            )
        except ValueError:
            QMessageBox.warning(
                self,
                "Contexto",
                "El contexto debe ser un número entero.",
            )
            return

        if context <= 0:
            QMessageBox.warning(
                self,
                "Contexto",
                "El contexto debe ser mayor que cero.",
            )
            return

        prompt = self.prompt_input.text().strip()

        if not prompt:
            QMessageBox.warning(
                self,
                "Prompt",
                "El prompt no puede estar vacío.",
            )
            return

        answer = QMessageBox.question(
            self,
            "Confirmar ejecución",
            (
                f"Ejecutar '{model_name}' con "
                f"contexto {context:,}."
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if answer != QMessageBox.Yes:
            return

        self.run_button.setEnabled(False)

        try:
            result = self.client.generate(
                model_name,
                prompt,
                context_length=context,
                keep_alive="10m",
            )

            response_text = result.get(
                "response",
                "",
            )

            QMessageBox.information(
                self,
                "Ejecución completada",
                (
                    f"Modelo: {model_name}\n"
                    f"Contexto: {context:,}\n\n"
                    f"Respuesta:\n{response_text}"
                ),
            )

            self._refresh_running_models_only()

        except Exception as exc:
            QMessageBox.critical(
                self,
                "Error de Ollama",
                (
                    "No se pudo ejecutar el modelo.\n"
                    + self._format_ollama_error(exc)
                ),
            )

        finally:
            self.run_button.setEnabled(True)

    def _pull_model(self):
        model_name, accepted = QInputDialog.getText(
            self,
            "Descargar modelo",
            "Nombre del modelo de Ollama:",
        )

        if not accepted:
            return

        model_name = model_name.strip()

        if not model_name:
            QMessageBox.warning(
                self,
                "Modelo",
                "Debes introducir un nombre de modelo.",
            )
            return

        answer = QMessageBox.question(
            self,
            "Confirmar descarga",
            (
                f"Descargar el modelo '{model_name}' "
                "desde Ollama?"
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if answer != QMessageBox.Yes:
            return

        self.pull_button.setEnabled(False)
        self.pull_progress.setVisible(True)
        self.pull_progress.setValue(0)

        self.pull_thread = QThread(self)
        self.pull_worker = PullWorker(
            self.client,
            model_name,
        )
        self.pull_worker.moveToThread(
            self.pull_thread
        )

        self.pull_thread.started.connect(
            self.pull_worker.run
        )
        self.pull_worker.progress.connect(
            self._pull_progress
        )
        self.pull_worker.finished.connect(
            self._pull_finished
        )
        self.pull_worker.error.connect(
            self._pull_error
        )

        self.pull_worker.finished.connect(
            self.pull_thread.quit
        )
        self.pull_worker.error.connect(
            self.pull_thread.quit
        )
        self.pull_thread.finished.connect(
            self.pull_worker.deleteLater
        )
        self.pull_thread.finished.connect(
            self.pull_thread.deleteLater
        )

        self.pull_thread.start()

    def _pull_progress(self, data):
        status = str(data.get("status", ""))

        total = int(data.get("total", 0) or 0)
        completed = int(data.get("completed", 0) or 0)

        if total > 0:
            percent = int(
                completed * 100 / total
            )
            self.pull_progress.setValue(
                max(0, min(percent, 100))
            )

        if status:
            self.status_label.setText(
                f"Estado: {status}"
            )

    def _pull_finished(self, data):
        self.pull_progress.setValue(100)
        self.pull_progress.setVisible(False)
        self.pull_button.setEnabled(True)

        status = str(
            data.get("status", "completado")
        )

        self.status_label.setText(
            f"Estado: descarga completada ({status})"
        )

        QMessageBox.information(
            self,
            "Descarga completada",
            "El modelo se ha descargado correctamente.",
        )

        self.refresh()

    def _pull_error(self, message):
        self.pull_progress.setVisible(False)
        self.pull_button.setEnabled(True)

        self.status_label.setText(
            "Estado: error durante la descarga"
        )

        QMessageBox.critical(
            self,
            "Error de descarga",
            (
                "No se pudo descargar el modelo.\n"
                f"{message}"
            ),
        )

    def _stop_selected_model(self):
        current = self.running_list.currentItem()

        if current is None:
            QMessageBox.warning(
                self,
                "Modelo",
                "Selecciona primero un modelo activo.",
            )
            return

        model_name = current.data(32)

        if not model_name:
            model_name = current.text().split(
                " — ",
                1,
            )[0].strip()

        if not model_name:
            QMessageBox.warning(
                self,
                "Modelo",
                "No se pudo identificar el modelo activo.",
            )
            return

        answer = QMessageBox.question(
            self,
            "Confirmar detención",
            (
                f"¿Descargar '{model_name}' de la memoria?\n\n"
                "El modelo no se eliminará del sistema."
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if answer != QMessageBox.Yes:
            return

        self.stop_button.setEnabled(False)

        try:
            self.client.stop_model(
                str(model_name)
            )

            QMessageBox.information(
                self,
                "Modelo detenido",
                (
                    f"'{model_name}' se ha descargado "
                    "de la memoria."
                ),
            )

            self._refresh_running_models_only()

        except Exception as exc:
            QMessageBox.critical(
                self,
                "Error de Ollama",
                (
                    "No se pudo detener el modelo.\n"
                    + self._format_ollama_error(exc)
                ),
            )

        finally:
            self.stop_button.setEnabled(True)

    def _delete_selected_model(self):
        model_name = self._selected_model_name()

        if not model_name:
            QMessageBox.warning(
                self,
                "Modelo",
                "Selecciona primero un modelo.",
            )
            return

        answer = QMessageBox.warning(
            self,
            "Confirmar eliminación",
            (
                f"¿Eliminar el modelo "
                f"'{model_name}'?\n\n"
                "Esta operación es destructiva."
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if answer != QMessageBox.Yes:
            return

        self.delete_button.setEnabled(False)

        try:
            self.client.delete_model(
                model_name
            )

            QMessageBox.information(
                self,
                "Modelo eliminado",
                f"Se ha eliminado '{model_name}'.",
            )

            self.refresh()

        except Exception as exc:
            QMessageBox.critical(
                self,
                "Error de Ollama",
                (
                    "No se pudo eliminar el modelo.\n"
                    + self._format_ollama_error(exc)
                ),
            )

        finally:
            self.delete_button.setEnabled(True)
