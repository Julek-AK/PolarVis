
# Builtins

# External Libraries
from PyQt6 import QtWidgets, QtCore
from numpy import degrees

# Internal Support


class PixelInfoPanel(QtWidgets.QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QGridLayout(self)

        self.intensity_label = QtWidgets.QLabel("Intensity: –")
        self.dolp_label = QtWidgets.QLabel("DoLP: –")
        self.theta_label = QtWidgets.QLabel("Angle: –")

        layout.addWidget(self.intensity_label, 0, 0)
        layout.addWidget(self.dolp_label, 1, 0)
        layout.addWidget(self.theta_label, 2, 0)

        layout.setContentsMargins(4, 4, 4, 4)

    def update_values(self, intensity: float, dolp: float, theta: float):
        theta_deg = degrees(theta)
        self.intensity_label.setText(f"Intensity: {intensity:.3f}")
        self.dolp_label.setText(f"DoLP: {dolp:.3f}")
        self.theta_label.setText(f"Angle: {theta_deg:.1f} deg")

        