"""
PyQt5-based professional GUI for the Macrophage vs Bacteria simulation.
Provides interactive controls, real-time visualization, and statistics display.
"""
import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QSlider, 
                             QGroupBox, QGridLayout, QTextEdit, QProgressBar)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPainter, QColor, QFont, QPen, QBrush
import numpy as np
import config


class GridCanvas(QWidget):
    """Custom widget for rendering the simulation grid."""
    
    def __init__(self, env, parent_gui=None):
        super().__init__()
        self.env = env
        self.parent_gui = parent_gui
        self.cell_size = 20
        self.setMinimumSize(env.size * self.cell_size, env.size * self.cell_size)
        self.movement_trail = []
        self.toxin_flash = False
        self.click_mode = None  # None, 'bacteria', or 'nutrient'
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw grid background
        painter.fillRect(0, 0, self.width(), self.height(), QColor(240, 240, 245))
        
        # Draw grid lines
        painter.setPen(QPen(QColor(200, 200, 200), 1))
        for i in range(self.env.size + 1):
            painter.drawLine(i * self.cell_size, 0, i * self.cell_size, self.height())
            painter.drawLine(0, i * self.cell_size, self.width(), i * self.cell_size)
        
        # Draw movement trail (fading)
        for i, (tx, ty) in enumerate(self.movement_trail):
            alpha = int(50 + (i / len(self.movement_trail)) * 150)
            painter.setBrush(QBrush(QColor(50, 100, 230, alpha)))
            painter.setPen(Qt.NoPen)
            x = tx * self.cell_size + self.cell_size // 4
            y = ty * self.cell_size + self.cell_size // 4
            painter.drawEllipse(x, y, self.cell_size // 2, self.cell_size // 2)
        
        # Draw nutrients (green circles with glow)
        for (nx, ny) in self.env.nutrients:
            x = nx * self.cell_size
            y = ny * self.cell_size
            # Glow effect
            painter.setBrush(QBrush(QColor(100, 255, 100, 80)))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(x - 3, y - 3, self.cell_size + 6, self.cell_size + 6)
            # Main nutrient
            painter.setBrush(QBrush(QColor(50, 220, 50)))
            painter.drawEllipse(x + 2, y + 2, self.cell_size - 4, self.cell_size - 4)
        
        # Draw bacteria (red with health indicator and special states)
        for b in self.env.bacteria:
            bx, by = b.position
            x = bx * self.cell_size
            y = by * self.cell_size
            
            # Biofilm glow effect (yellow aura)
            if hasattr(b, 'in_biofilm') and b.in_biofilm:
                painter.setBrush(QBrush(QColor(255, 255, 0, 60)))
                painter.setPen(Qt.NoPen)
                painter.drawEllipse(x - 4, y - 4, self.cell_size + 8, self.cell_size + 8)
            
            # Virulence glow effect (purple aura)
            if hasattr(b, 'virulence_active') and b.virulence_active:
                painter.setBrush(QBrush(QColor(148, 0, 211, 80)))
                painter.setPen(Qt.NoPen)
                painter.drawEllipse(x - 3, y - 3, self.cell_size + 6, self.cell_size + 6)
            
            # Health-based color (darker when weaker)
            health_ratio = b.health / 20.0
            red_value = int(150 + health_ratio * 105)
            
            # Immature bacteria are paler
            if hasattr(b, 'age') and b.age < config.BACTERIA_MATURITY_AGE:
                painter.setBrush(QBrush(QColor(red_value, 120, 120)))  # Paler red
                painter.setPen(QPen(QColor(100, 80, 80), 1))
            else:
                painter.setBrush(QBrush(QColor(red_value, 30, 30)))
                painter.setPen(QPen(QColor(100, 0, 0), 2))
            
            painter.drawEllipse(x + 1, y + 1, self.cell_size - 2, self.cell_size - 2)
            
            # Health bar above bacteria
            bar_width = int((self.cell_size - 4) * health_ratio)
            painter.fillRect(x + 2, y - 5, bar_width, 3, QColor(255, 0, 0))
        
        # Draw macrophage (blue with special effects)
        mx, my = self.env.macrophage.position
        x = mx * self.cell_size
        y = my * self.cell_size
        
        # Vision radius circle (cyan dashed)
        painter.setPen(QPen(QColor(0, 200, 255, 100), 2, Qt.DashLine))
        painter.setBrush(Qt.NoBrush)
        vision_radius = 6 * self.cell_size
        painter.drawEllipse(x + self.cell_size // 2 - vision_radius,
                          y + self.cell_size // 2 - vision_radius,
                          vision_radius * 2, vision_radius * 2)
        
        # Toxin radius (yellow)
        if self.toxin_flash:
            painter.setPen(QPen(QColor(255, 255, 0, 200), 3))
            painter.setBrush(QBrush(QColor(255, 255, 0, 50)))
            toxin_radius = 2 * self.cell_size
            painter.drawEllipse(x + self.cell_size // 2 - toxin_radius,
                              y + self.cell_size // 2 - toxin_radius,
                              toxin_radius * 2, toxin_radius * 2)
        
        # Macrophage body (bright blue with glow)
        painter.setBrush(QBrush(QColor(30, 100, 255, 180)))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(x - 2, y - 2, self.cell_size + 4, self.cell_size + 4)
        painter.setBrush(QBrush(QColor(50, 120, 255)))
        painter.setPen(QPen(QColor(0, 50, 200), 3))
        painter.drawEllipse(x + 1, y + 1, self.cell_size - 2, self.cell_size - 2)
        
        # Health indicator
        health_ratio = self.env.macrophage.health / 100.0
        bar_width = int((self.cell_size - 4) * health_ratio)
        painter.fillRect(x + 2, y - 6, bar_width, 4, QColor(0, 200, 255))
    
    def update_display(self, env):
        """Update the environment and redraw."""
        self.env = env
        # Update movement trail
        mx, my = env.macrophage.position
        self.movement_trail.append((mx, my))
        if len(self.movement_trail) > 20:
            self.movement_trail.pop(0)
        self.update()
    
    def mousePressEvent(self, event):
        """Handle mouse clicks on the grid."""
        if self.click_mode is None:
            return
        
        # Convert pixel coordinates to grid coordinates
        grid_x = event.x() // self.cell_size
        grid_y = event.y() // self.cell_size
        
        # Validate coordinates
        if grid_x < 0 or grid_x >= self.env.size or grid_y < 0 or grid_y >= self.env.size:
            return
        
        # Check if cell is already occupied
        mx, my = self.env.macrophage.position
        if (grid_x, grid_y) == (mx, my):
            if self.parent_gui:
                self.parent_gui.log_message(f"❌ Cannot place at ({grid_x}, {grid_y}) - Macrophage is there!")
            return
        
        # Check if bacteria exists at position
        bacteria_at_pos = any(b.position == (grid_x, grid_y) for b in self.env.bacteria)
        
        if self.click_mode == 'bacteria':
            if bacteria_at_pos:
                if self.parent_gui:
                    self.parent_gui.log_message(f"❌ Bacteria already exists at ({grid_x}, {grid_y})")
                return
            
            # Add new bacteria
            from simulator.entities import Bacteria
            new_bacteria = Bacteria(position=(grid_x, grid_y), health=config.BACTERIA_MAX_HEALTH)
            self.env.bacteria.append(new_bacteria)
            if self.parent_gui:
                self.parent_gui.log_message(f"🦠 Added bacteria at ({grid_x}, {grid_y})")
                self.parent_gui.update_stats()
        
        elif self.click_mode == 'nutrient':
            if (grid_x, grid_y) in self.env.nutrients:
                if self.parent_gui:
                    self.parent_gui.log_message(f"❌ Nutrient already exists at ({grid_x}, {grid_y})")
                return
            
            if bacteria_at_pos:
                if self.parent_gui:
                    self.parent_gui.log_message(f"❌ Cannot place nutrient at ({grid_x}, {grid_y}) - Bacteria is there!")
                return
            
            # Add new nutrient
            self.env.nutrients.add((grid_x, grid_y))
            if self.parent_gui:
                self.parent_gui.log_message(f"🟢 Added nutrient at ({grid_x}, {grid_y})")
                self.parent_gui.update_stats()
        
        # Redraw canvas
        self.update()


class SimulationGUI(QMainWindow):
    """Main PyQt5 window for the simulation."""
    
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.original_env = None  # For reset
        self.running = False
        self.speed = 50  # milliseconds per step
        
        self.init_ui()
        self.timer = QTimer()
        self.timer.timeout.connect(self.step_simulation)
        
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("🦠 Macrophage vs Bacteria Simulation 💉")
        self.setGeometry(100, 100, 1200, 700)
        
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)
        
        # Left side: Grid canvas
        self.canvas = GridCanvas(self.env, parent_gui=self)
        main_layout.addWidget(self.canvas, stretch=2)
        
        # Right side: Control panel
        control_panel = QWidget()
        control_layout = QVBoxLayout()
        control_panel.setLayout(control_layout)
        main_layout.addWidget(control_panel, stretch=1)
        
        # === Title ===
        title = QLabel("🎮 Simulation Control")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        control_layout.addWidget(title)
        
        # === Control Buttons ===
        button_group = QGroupBox("Controls")
        button_layout = QGridLayout()
        button_group.setLayout(button_layout)
        
        self.start_btn = QPushButton("▶️ Start")
        self.start_btn.clicked.connect(self.start_simulation)
        button_layout.addWidget(self.start_btn, 0, 0)
        
        self.pause_btn = QPushButton("⏸️ Pause")
        self.pause_btn.clicked.connect(self.pause_simulation)
        self.pause_btn.setEnabled(False)
        button_layout.addWidget(self.pause_btn, 0, 1)
        
        self.step_btn = QPushButton("⏭️ Step")
        self.step_btn.clicked.connect(self.single_step)
        button_layout.addWidget(self.step_btn, 1, 0)
        
        self.reset_btn = QPushButton("🔄 Reset")
        self.reset_btn.clicked.connect(self.reset_simulation)
        button_layout.addWidget(self.reset_btn, 1, 1)
        
        control_layout.addWidget(button_group)
        
        # === Manual Placement ===
        placement_group = QGroupBox("🖱️ Manual Placement")
        placement_layout = QVBoxLayout()
        placement_group.setLayout(placement_layout)
        
        placement_info = QLabel("Click on grid to place:")
        placement_info.setStyleSheet("font-size: 11px; color: gray;")
        placement_layout.addWidget(placement_info)
        
        self.add_bacteria_btn = QPushButton("🦠 Add Bacteria")
        self.add_bacteria_btn.setCheckable(True)
        self.add_bacteria_btn.clicked.connect(self.toggle_bacteria_mode)
        self.add_bacteria_btn.setStyleSheet("""
            QPushButton:checked { background-color: #FF6B6B; color: white; font-weight: bold; }
        """)
        placement_layout.addWidget(self.add_bacteria_btn)
        
        self.add_nutrient_btn = QPushButton("🟢 Add Nutrient")
        self.add_nutrient_btn.setCheckable(True)
        self.add_nutrient_btn.clicked.connect(self.toggle_nutrient_mode)
        self.add_nutrient_btn.setStyleSheet("""
            QPushButton:checked { background-color: #51CF66; color: white; font-weight: bold; }
        """)
        placement_layout.addWidget(self.add_nutrient_btn)
        
        self.mode_status = QLabel("Mode: None")
        self.mode_status.setAlignment(Qt.AlignCenter)
        self.mode_status.setStyleSheet("font-weight: bold; padding: 5px;")
        placement_layout.addWidget(self.mode_status)
        
        control_layout.addWidget(placement_group)
        
        # === Speed Control ===
        speed_group = QGroupBox("Speed Control")
        speed_layout = QVBoxLayout()
        speed_group.setLayout(speed_layout)
        
        self.speed_label = QLabel(f"Speed: {1000 // self.speed} steps/sec")
        self.speed_label.setAlignment(Qt.AlignCenter)
        speed_layout.addWidget(self.speed_label)
        
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(1)
        self.speed_slider.setMaximum(100)
        self.speed_slider.setValue(50)
        self.speed_slider.valueChanged.connect(self.update_speed)
        speed_layout.addWidget(self.speed_slider)
        
        speed_info = QLabel("← Faster | Slower →")
        speed_info.setAlignment(Qt.AlignCenter)
        speed_info.setStyleSheet("color: gray; font-size: 10px;")
        speed_layout.addWidget(speed_info)
        
        control_layout.addWidget(speed_group)
        
        # === Statistics ===
        stats_group = QGroupBox("📊 Live Statistics")
        stats_layout = QVBoxLayout()
        stats_group.setLayout(stats_layout)
        
        self.step_label = QLabel("Step: 0")
        stats_layout.addWidget(self.step_label)
        
        self.mac_health_label = QLabel("Macrophage Health: 100/100")
        stats_layout.addWidget(self.mac_health_label)
        
        self.mac_health_bar = QProgressBar()
        self.mac_health_bar.setMaximum(100)
        self.mac_health_bar.setValue(100)
        self.mac_health_bar.setStyleSheet("""
            QProgressBar::chunk { background-color: #3264FF; }
        """)
        stats_layout.addWidget(self.mac_health_bar)
        
        self.bacteria_label = QLabel("Bacteria Count: 5")
        stats_layout.addWidget(self.bacteria_label)
        
        self.bacteria_bar = QProgressBar()
        self.bacteria_bar.setMaximum(40)
        self.bacteria_bar.setValue(5)
        self.bacteria_bar.setStyleSheet("""
            QProgressBar::chunk { background-color: #FF3232; }
        """)
        stats_layout.addWidget(self.bacteria_bar)
        
        self.nutrient_label = QLabel("Nutrients: 10")
        stats_layout.addWidget(self.nutrient_label)
        
        # New biological stats
        self.biofilm_label = QLabel("Biofilm Bacteria: 0")
        self.biofilm_label.setStyleSheet("font-size: 10px; color: #FF9800;")
        stats_layout.addWidget(self.biofilm_label)
        
        self.virulent_label = QLabel("Virulent Bacteria: 0")
        self.virulent_label.setStyleSheet("font-size: 10px; color: #9C27B0;")
        stats_layout.addWidget(self.virulent_label)
        
        self.exhaustion_label = QLabel("Macrophage Exhaustion: 0")
        self.exhaustion_label.setStyleSheet("font-size: 10px; color: #2196F3;")
        stats_layout.addWidget(self.exhaustion_label)
        
        control_layout.addWidget(stats_group)
        
        # === Status Log ===
        log_group = QGroupBox("📝 Status Log")
        log_layout = QVBoxLayout()
        log_group.setLayout(log_layout)
        
        self.status_log = QTextEdit()
        self.status_log.setReadOnly(True)
        self.status_log.setMaximumHeight(150)
        self.status_log.setStyleSheet("background-color: #F5F5F5; font-family: monospace;")
        log_layout.addWidget(self.status_log)
        
        control_layout.addWidget(log_group)
        
        # === Winner Display ===
        self.winner_label = QLabel("")
        self.winner_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.winner_label.setAlignment(Qt.AlignCenter)
        self.winner_label.setStyleSheet("color: green; padding: 10px;")
        control_layout.addWidget(self.winner_label)
        
        control_layout.addStretch()
        
        # Initial log
        self.log_message("Simulation initialized. Press Start to begin.")
        self.update_stats()
    
    def start_simulation(self):
        """Start the simulation timer."""
        self.running = True
        self.timer.start(self.speed)
        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.log_message("▶️ Simulation started!")
    
    def pause_simulation(self):
        """Pause the simulation."""
        self.running = False
        self.timer.stop()
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.log_message("⏸️ Simulation paused.")
    
    def single_step(self):
        """Execute a single simulation step."""
        if not self.env.done:
            self.env.step()
            self.canvas.update_display(self.env)
            self.update_stats()
            self.check_end_condition()
    
    def step_simulation(self):
        """Timer callback for automatic stepping."""
        if self.running and not self.env.done:
            self.single_step()
        elif self.env.done:
            self.pause_simulation()
    
    def reset_simulation(self):
        """Reset the simulation to initial state."""
        from simulator.environment import Environment
        self.pause_simulation()
        self.env = Environment()
        self.canvas.update_display(self.env)
        self.canvas.movement_trail = []
        self.winner_label.setText("")
        self.update_stats()
        self.log_message("🔄 Simulation reset to initial state.")
    
    def update_speed(self, value):
        """Update simulation speed from slider."""
        # Map slider (1-100) to speed (200ms - 10ms)
        self.speed = max(10, 210 - value * 2)
        if self.running:
            self.timer.setInterval(self.speed)
        self.speed_label.setText(f"Speed: {1000 // self.speed} steps/sec")
    
    def update_stats(self):
        """Update statistics display."""
        self.step_label.setText(f"Step: {self.env.step_count}")
        
        mac_health = self.env.macrophage.health
        self.mac_health_label.setText(f"Macrophage Health: {mac_health}/100")
        self.mac_health_bar.setValue(mac_health)
        
        bacteria_count = len(self.env.bacteria)
        self.bacteria_label.setText(f"Bacteria Count: {bacteria_count}")
        self.bacteria_bar.setValue(bacteria_count)
        
        nutrient_count = len(self.env.nutrients)
        self.nutrient_label.setText(f"Nutrients: {nutrient_count}")
        
        # Update biological stats
        biofilm_count = sum(1 for b in self.env.bacteria if hasattr(b, 'in_biofilm') and b.in_biofilm)
        self.biofilm_label.setText(f"Biofilm Bacteria: {biofilm_count}")
        
        virulent_count = sum(1 for b in self.env.bacteria if hasattr(b, 'virulence_active') and b.virulence_active)
        self.virulent_label.setText(f"Virulent Bacteria: {virulent_count}")
        
        exhaustion = getattr(self.env.macrophage, 'exhaustion', 0)
        self.exhaustion_label.setText(f"Macrophage Exhaustion: {exhaustion}")
        
        # Warnings
        if mac_health < 30 and mac_health > 0:
            self.log_message("⚠️ WARNING: Macrophage health critical!")
        if bacteria_count > 30:
            self.log_message("☠️ WARNING: Bacteria outbreak!")
        if biofilm_count >= 3:
            self.log_message("🟡 Biofilm detected! Bacteria are protected!")
        if virulent_count > 0:
            self.log_message("🟣 Virulent bacteria actively producing toxins!")
    
    def check_end_condition(self):
        """Check if simulation has ended."""
        if self.env.done:
            winner_text = f"🏆 {self.env.winner} WINS!\n{self.env.win_reason}"
            self.winner_label.setText(winner_text)
            self.log_message(f"🏁 GAME OVER: {self.env.winner} wins! ({self.env.win_reason})")
            self.pause_simulation()
    
    def toggle_bacteria_mode(self):
        """Toggle bacteria placement mode."""
        if self.add_bacteria_btn.isChecked():
            self.canvas.click_mode = 'bacteria'
            self.add_nutrient_btn.setChecked(False)
            self.mode_status.setText("Mode: Add Bacteria 🦠")
            self.mode_status.setStyleSheet("font-weight: bold; padding: 5px; background-color: #FFE0E0;")
            self.log_message("🖱️ Click on grid to add bacteria")
        else:
            self.canvas.click_mode = None
            self.mode_status.setText("Mode: None")
            self.mode_status.setStyleSheet("font-weight: bold; padding: 5px;")
    
    def toggle_nutrient_mode(self):
        """Toggle nutrient placement mode."""
        if self.add_nutrient_btn.isChecked():
            self.canvas.click_mode = 'nutrient'
            self.add_bacteria_btn.setChecked(False)
            self.mode_status.setText("Mode: Add Nutrient 🟢")
            self.mode_status.setStyleSheet("font-weight: bold; padding: 5px; background-color: #E0FFE0;")
            self.log_message("🖱️ Click on grid to add nutrients")
        else:
            self.canvas.click_mode = None
            self.mode_status.setText("Mode: None")
            self.mode_status.setStyleSheet("font-weight: bold; padding: 5px;")
    
    def log_message(self, message):
        """Add a message to the status log."""
        self.status_log.append(f"[Step {self.env.step_count}] {message}")
        # Auto-scroll to bottom
        self.status_log.verticalScrollBar().setValue(
            self.status_log.verticalScrollBar().maximum()
        )


def launch_gui(env):
    """
    Launch the PyQt5 GUI for the simulation.
    
    Args:
        env: The simulation environment to visualize
    """
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern style
    gui = SimulationGUI(env)
    gui.show()
    sys.exit(app.exec_())


