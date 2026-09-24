from PySide6.QtWidgets import (
    QLabel,
    QMainWindow,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from hardware_detector import HardwareDetector
from core.knowledge_base import KnowledgeBase
from core.recommendation_engine import RecommendationEngine
from ui.hardware_page import HardwarePage
from ui.ollama_page import OllamaPage


class MainWindow(QMainWindow):
    """Ventana principal de Ollama Manager."""

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Ollama Manager")
        self.resize(1400, 900)

        detector = HardwareDetector()
        self.hardware = detector.detect_normalized()

        knowledge_base = KnowledgeBase(
            "knowledge/hardware_profiles.json"
        )
        knowledge_base.load()

        recommendation_engine = RecommendationEngine(
            knowledge_base
        )
        self.recommendations = (
            recommendation_engine.analyze(self.hardware)
        )

        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)

        title = QLabel("Ollama Manager")
        title.setStyleSheet(
            "font-size: 28px; "
            "font-weight: bold; "
            "padding: 20px;"
        )

        self.pages = QTabWidget()

        self.hardware_page = HardwarePage(
            self.hardware,
            self.recommendations,
        )

        self.ollama_page = OllamaPage()

        self.pages.addTab(self.hardware_page, "Hardware")
        self.pages.addTab(self.ollama_page, "Ollama")

        layout.addWidget(title)
        layout.addWidget(self.pages)

        self.setCentralWidget(central_widget)
