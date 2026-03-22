"""Research-style PyQt5 GUI for Pathway 1 environment."""

import sys

import numpy as np
import config
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QFont, QPainter, QPen
from PyQt5.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)


class GridCanvas(QWidget):
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.cell_size = 24
        self.setMinimumSize(env.width * self.cell_size, env.height * self.cell_size)
        self.setStyleSheet("background-color: #f4f7fb;")

    @staticmethod
    def _blend(c1, c2, alpha):
        alpha = max(0.0, min(1.0, alpha))
        return QColor(
            int(c1.red() * (1 - alpha) + c2.red() * alpha),
            int(c1.green() * (1 - alpha) + c2.green() * alpha),
            int(c1.blue() * (1 - alpha) + c2.blue() * alpha),
        )

    def _draw_cell_background(self, painter, x, y):
        if (x, y) in self.env.blocked_tiles:
            painter.fillRect(
                x * self.cell_size,
                y * self.cell_size,
                self.cell_size,
                self.cell_size,
                QColor("#2e3440"),
            )
            return

        nutrient = float(self.env.nutrients[y, x])
        nutrient_norm = min(1.0, nutrient / max(config.PATCH_NUTRIENT_CAPACITY, 1e-6))
        low = QColor("#eef4ec")
        high = QColor("#3d9f50")
        nutrient_color = self._blend(low, high, nutrient_norm)
        painter.fillRect(
            x * self.cell_size,
            y * self.cell_size,
            self.cell_size,
            self.cell_size,
            nutrient_color,
        )

        chem = float(self.env.chemokine[y, x])
        chem_max = float(np.max(self.env.chemokine))
        if chem_max > 1e-9:
            chem_alpha = min(0.40, 0.40 * (chem / chem_max))
            chem_color = QColor(220, 38, 38, int(255 * chem_alpha))
            painter.fillRect(
                x * self.cell_size,
                y * self.cell_size,
                self.cell_size,
                self.cell_size,
                chem_color,
            )

        if (x, y) in self.env.surface_patches:
            marker = QColor("#f8fafc")
            marker.setAlpha(120)
            cx = x * self.cell_size + (self.cell_size // 2) - 2
            cy = y * self.cell_size + (self.cell_size // 2) - 2
            painter.setPen(Qt.NoPen)
            painter.setBrush(marker)
            painter.drawEllipse(cx, cy, 4, 4)

    def _draw_entities(self, painter):
        for n in self.env.neutrophils:
            x, y = n.position
            cx = x * self.cell_size + self.cell_size // 2
            cy = y * self.cell_size + self.cell_size // 2
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor("#0ea5e9"))
            painter.drawEllipse(cx - 5, cy - 5, 10, 10)

        for b in self.env.bacteria:
            x, y = b.position
            cx = x * self.cell_size + self.cell_size // 2
            cy = y * self.cell_size + self.cell_size // 2
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor("#dc2626"))
            painter.drawEllipse(cx - 5, cy - 5, 10, 10)

        mx, my = self.env.macrophage.position
        cx = mx * self.cell_size + self.cell_size // 2
        cy = my * self.cell_size + self.cell_size // 2
        painter.setPen(QPen(QColor("#1d4ed8"), 2))
        painter.setBrush(QColor("#60a5fa"))
        painter.drawEllipse(cx - 7, cy - 7, 14, 14)

    def _draw_grid_lines(self, painter):
        painter.setPen(QPen(QColor("#d6dee7"), 1))
        for x in range(self.env.width + 1):
            painter.drawLine(
                x * self.cell_size,
                0,
                x * self.cell_size,
                self.env.height * self.cell_size,
            )
        for y in range(self.env.height + 1):
            painter.drawLine(
                0,
                y * self.cell_size,
                self.env.width * self.cell_size,
                y * self.cell_size,
            )

    def paintEvent(self, _):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)

        for y in range(self.env.height):
            for x in range(self.env.width):
                self._draw_cell_background(painter, x, y)

        self._draw_grid_lines(painter)
        painter.setRenderHint(QPainter.Antialiasing, True)
        self._draw_entities(painter)

    def update_display(self, env):
        self.env = env
        self.update()


class SimulationGUI(QMainWindow):
    def __init__(self, env, policy_agent=None):
        super().__init__()
        self.env = env
        self.policy_agent = policy_agent
        self.timer = QTimer()
        self.timer.timeout.connect(self.step_simulation)
        self.interval = 80

        self.setWindowTitle("Pathway 1 - Innate Response Research Console")
        self.setGeometry(100, 100, 1380, 820)

        self.setStyleSheet(
            """
            QMainWindow {
                background: #f1f5f9;
            }
            QLabel#title {
                color: #0f172a;
                font-size: 23px;
                font-weight: 700;
            }
            QLabel#subtitle {
                color: #475569;
                font-size: 12px;
            }
            QFrame#panel {
                background: #ffffff;
                border: 1px solid #dbe3ec;
                border-radius: 10px;
            }
            QLabel.section {
                color: #1e293b;
                font-size: 14px;
                font-weight: 700;
            }
            QLabel.metricLabel {
                color: #64748b;
                font-size: 11px;
            }
            QLabel.metricValue {
                color: #0f172a;
                font-size: 18px;
                font-weight: 700;
            }
            QPushButton {
                background: #0f6b4b;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 12px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #0d5c40;
            }
            QPushButton#secondary {
                background: #e2e8f0;
                color: #0f172a;
            }
            QPushButton#secondary:hover {
                background: #cfd8e3;
            }
            QSlider::groove:horizontal {
                border: 0;
                height: 6px;
                background: #cbd5e1;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #0f6b4b;
                width: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
            """
        )

        app_font = QFont("Segoe UI", 10)
        self.setFont(app_font)

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(16)

        left = QVBoxLayout()
        left.setSpacing(12)
        root.addLayout(left, stretch=7)

        title = QLabel("Pathway 1: Acute Alveolar Hotspot")
        title.setObjectName("title")
        subtitle = QLabel(
            "Compartmental bacterial invasion with delayed neutrophil recruitment and chemokine diffusion"
        )
        subtitle.setObjectName("subtitle")
        left.addWidget(title)
        left.addWidget(subtitle)

        canvas_panel = QFrame()
        canvas_panel.setObjectName("panel")
        canvas_layout = QVBoxLayout(canvas_panel)
        canvas_layout.setContentsMargins(12, 12, 12, 12)
        canvas_layout.setSpacing(10)

        self.canvas = GridCanvas(env)
        canvas_layout.addWidget(self.canvas)
        left.addWidget(canvas_panel)

        legend_panel = QFrame()
        legend_panel.setObjectName("panel")
        legend_layout = QHBoxLayout(legend_panel)
        legend_layout.setContentsMargins(12, 10, 12, 10)
        legend_layout.setSpacing(14)
        legend_layout.addWidget(self._legend_item("#2e3440", "Compartment Wall / Bottleneck"))
        legend_layout.addWidget(self._legend_item("#3d9f50", "Nutrient-Rich Tissue"))
        legend_layout.addWidget(self._legend_item("#dc2626", "Bacteria"))
        legend_layout.addWidget(self._legend_item("#60a5fa", "Macrophage"))
        legend_layout.addWidget(self._legend_item("#0ea5e9", "Neutrophil"))
        legend_layout.addWidget(self._legend_item("#ef4444", "Chemokine Overlay"))
        legend_layout.addStretch()
        left.addWidget(legend_panel)

        side = QVBoxLayout()
        side.setSpacing(12)
        root.addLayout(side, stretch=4)

        metrics_panel = QFrame()
        metrics_panel.setObjectName("panel")
        metrics_layout = QGridLayout(metrics_panel)
        metrics_layout.setContentsMargins(12, 12, 12, 12)
        metrics_layout.setHorizontalSpacing(14)
        metrics_layout.setVerticalSpacing(10)

        section_metrics = QLabel("Runtime Metrics")
        section_metrics.setProperty("class", "section")
        metrics_layout.addWidget(section_metrics, 0, 0, 1, 2)

        self.metric_step = self._metric_block("Step")
        self.metric_phase = self._metric_block("Phase")
        self.metric_bacteria = self._metric_block("Bacteria")
        self.metric_neutrophils = self._metric_block("Neutrophils")
        self.metric_mhealth = self._metric_block("Macrophage Health")
        self.metric_damage = self._metric_block("Tissue Damage")
        self.metric_utility = self._metric_block("Host Utility")
        self.metric_chem = self._metric_block("Chemokine Peak")

        metric_widgets = [
            self.metric_step,
            self.metric_phase,
            self.metric_bacteria,
            self.metric_neutrophils,
            self.metric_mhealth,
            self.metric_damage,
            self.metric_utility,
            self.metric_chem,
        ]
        for idx, box in enumerate(metric_widgets, start=1):
            row = ((idx - 1) // 2) + 1
            col = (idx - 1) % 2
            metrics_layout.addWidget(box, row, col)

        side.addWidget(metrics_panel)

        controls_panel = QFrame()
        controls_panel.setObjectName("panel")
        controls_layout = QVBoxLayout(controls_panel)
        controls_layout.setContentsMargins(12, 12, 12, 12)
        controls_layout.setSpacing(10)
        section_control = QLabel("Experiment Controls")
        section_control.setProperty("class", "section")
        controls_layout.addWidget(section_control)

        btn_start = QPushButton("Start Continuous Run")
        btn_start.clicked.connect(self.start)
        controls_layout.addWidget(btn_start)

        btn_pause = QPushButton("Pause")
        btn_pause.setObjectName("secondary")
        btn_pause.clicked.connect(self.pause)
        controls_layout.addWidget(btn_pause)

        btn_step = QPushButton("Advance Single Step")
        btn_step.setObjectName("secondary")
        btn_step.clicked.connect(self.step_once)
        controls_layout.addWidget(btn_step)

        speed_label = QLabel("Simulation Interval (ms)")
        speed_label.setProperty("class", "metricLabel")
        controls_layout.addWidget(speed_label)

        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(20)
        slider.setMaximum(500)
        slider.setSingleStep(10)
        slider.setValue(self.interval)
        slider.valueChanged.connect(self.set_interval)
        controls_layout.addWidget(slider)

        self.interval_value = QLabel(f"{self.interval} ms")
        self.interval_value.setProperty("class", "metricValue")
        controls_layout.addWidget(self.interval_value)

        side.addWidget(controls_panel)

        notes_panel = QFrame()
        notes_panel.setObjectName("panel")
        notes_layout = QVBoxLayout(notes_panel)
        notes_layout.setContentsMargins(12, 12, 12, 12)
        notes_layout.setSpacing(8)
        notes_title = QLabel("Simulation Notes")
        notes_title.setProperty("class", "section")
        notes_layout.addWidget(notes_title)

        self.status = QLabel()
        self.status.setProperty("class", "metricLabel")
        self.status.setWordWrap(True)
        notes_layout.addWidget(self.status)
        side.addWidget(notes_panel)

        side.addStretch()
        self.refresh_status()

    def _legend_item(self, color, text):
        wrap = QWidget()
        layout = QHBoxLayout(wrap)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        swatch = QLabel()
        swatch.setFixedSize(14, 14)
        swatch.setStyleSheet(f"background: {color}; border: 1px solid #cbd5e1; border-radius: 3px;")

        label = QLabel(text)
        label.setProperty("class", "metricLabel")

        layout.addWidget(swatch)
        layout.addWidget(label)
        return wrap

    def _metric_block(self, title):
        block = QFrame()
        block.setObjectName("panel")
        block.setStyleSheet("QFrame#panel { border-radius: 8px; }")
        layout = QVBoxLayout(block)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(2)

        label = QLabel(title)
        label.setProperty("class", "metricLabel")
        value = QLabel("--")
        value.setProperty("class", "metricValue")
        block.value_label = value

        layout.addWidget(label)
        layout.addWidget(value)
        return block

    def set_interval(self, value):
        self.interval = int(value)
        self.interval_value.setText(f"{self.interval} ms")
        if self.timer.isActive():
            self.timer.setInterval(self.interval)

    def start(self):
        self.timer.start(self.interval)

    def pause(self):
        self.timer.stop()

    def step_once(self):
        if not self.env.done:
            if self.policy_agent is None:
                self.env.step()
            else:
                action = self.policy_agent.choose_action(self.env)
                self.env.step(macrophage_action=action)
            self.canvas.update_display(self.env)
            self.refresh_status()

    def step_simulation(self):
        if self.env.done:
            self.pause()
            return
        self.step_once()

    def refresh_status(self):
        bacteria_count = len(self.env.bacteria)
        neutro_count = len(self.env.neutrophils)
        utility = self.env.compute_host_utility()
        chem_peak = float(np.max(self.env.chemokine))

        self.metric_step.value_label.setText(str(self.env.step_count))
        self.metric_phase.value_label.setText(self.env.current_phase)
        self.metric_bacteria.value_label.setText(str(bacteria_count))
        self.metric_neutrophils.value_label.setText(str(neutro_count))
        self.metric_mhealth.value_label.setText(str(self.env.macrophage.health))
        self.metric_damage.value_label.setText(f"{self.env.tissue_damage:.2f}")
        self.metric_utility.value_label.setText(f"{utility:.2f}")
        self.metric_chem.value_label.setText(f"{chem_peak:.2f}")

        lines = [
            "Interpretation:",
            "- Dark cells are structural compartment walls; openings are bottleneck passages.",
            "- Green intensity shows local nutrient availability in tissue.",
            "- Red wash indicates chemokine concentration hot spots.",
        ]
        if self.env.done:
            self.metric_phase.value_label.setText("completed")
            lines.append(f"Winner: {self.env.winner}")
            lines.append(f"Reason: {self.env.win_reason}")

        self.status.setText("\n".join(lines))


def launch_gui(env, policy_agent=None):
    app = QApplication(sys.argv)
    gui = SimulationGUI(env, policy_agent=policy_agent)
    gui.show()
    sys.exit(app.exec_())


