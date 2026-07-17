
# Builtins

# External Libraries
from PyQt6 import QtWidgets, QtCore, QtGui
from numpy import degrees
import numpy as np

# Internal Support
from ..app.config.settings import settings


INTENSITY_CMAP = settings.get('visualization.colormaps.intensity')
DOLP_CMAP = settings.get('visualization.colormaps.dolp')
AOP_CMAP = settings.get('visualization.colormaps.aop')


class PixelInfoPanel(QtWidgets.QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QGridLayout(self)

        title = QtWidgets.QLabel("Pixel Information")
        title_font = QtGui.QFont()
        title_font.setBold(True)
        title_font.setPointSize(11)
        title.setFont(title_font)

        self.intensity_label = QtWidgets.QLabel("Intensity:")
        self.dolp_label = QtWidgets.QLabel("Degree of Linear Polarization:")
        self.theta_label = QtWidgets.QLabel("Polarization Angle:")

        self.intensity_scale = ColorScaleWidget(INTENSITY_CMAP, 0, 255)
        self.dolp_scale = ColorScaleWidget(DOLP_CMAP, 0, 1)
        self.theta_scale = ColorScaleWidget(AOP_CMAP, 0, 180)
        self.theta_display = AngleWidget()

        layout.addWidget(title, 0, 0)

        layout.addWidget(self.intensity_label, 1, 0)
        layout.addWidget(self.intensity_scale, 2, 0)

        layout.addWidget(self.dolp_label, 3, 0)
        layout.addWidget(self.dolp_scale, 4, 0)

        layout.addWidget(self.theta_label, 5, 0)
        layout.addWidget(self.theta_scale, 6, 0)
        layout.addWidget(self.theta_display, 7, 0)
        layout.setContentsMargins(4, 4, 4, 4)

    def update_values(self, intensity: float, dolp: float, theta: float):
        theta = np.mod(theta, np.pi)
        
        # Update labels
        self.intensity_label.setText(f"Intensity: {255*intensity}")
        self.dolp_label.setText(f"Degree of Linear Polarization: {dolp:.3f}")
        self.theta_label.setText(f"Polarization Angle: {degrees(theta):.1f} deg")

        # Update visuals
        self.intensity_scale.set_value(intensity * 255)
        self.dolp_scale.set_value(dolp)
        self.theta_scale.set_value(degrees(theta))
        self.theta_display.set_angle(theta)


class ColorScaleWidget(QtWidgets.QWidget):
    """Dynamic color scale for representing image parameters"""
    def __init__(self, cmap_name, vmin=0.0, vmax=1.0, parent=None, is_intensity=False):
        super().__init__(parent)
        self.cmap_name = cmap_name
        self.vmin = vmin
        self.vmax = vmax
        self.value = None

        # self.is_intensity = is_intensity  # TODO something isn't working with the intensity scale mismatch handling

        self._gradient_pixmap = None  # cache
        self.setMinimumHeight(30)

    def set_value(self, value):
        # if self.is_intensity:  # correct for intensity scale mismatch
        #     value /= 2

        if self.value == value:
            return
        self.value = value
        self.update()  # only overlay redraw

    def resizeEvent(self, event):
        self._generate_gradient()
        super().resizeEvent(event)

    def _generate_gradient(self):
        """Generate and cache gradient pixmap"""
        from matplotlib import cm

        w = self.width()
        h = self.height()
        if w <= 0 or h <= 0:
            return

        cmap = cm.get_cmap(self.cmap_name)

        gradient = np.linspace(0, 1, w)
        gradient = np.tile(gradient, (h, 1))

        rgba = cmap(gradient)
        rgba = (rgba * 255).astype(np.uint8)

        image = QtGui.QImage(
            rgba.data, w, h, QtGui.QImage.Format.Format_RGBA8888
        )

        self._gradient_pixmap = QtGui.QPixmap.fromImage(image.copy())

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)

        # Draw cached gradient
        if self._gradient_pixmap:
            painter.drawPixmap(self.rect(), self._gradient_pixmap)

        # Draw marker
        if self.value is not None:
            rect = self.rect()
            norm = (self.value - self.vmin) / (self.vmax - self.vmin)
            norm = np.clip(norm, 0.0, 1.0)

            x = int(norm * rect.width())

            painter.setPen(QtGui.QPen(QtGui.QColor("white"), 2))
            painter.drawLine(x, 0, x, rect.height())

            painter.drawText(x + 5, rect.height() // 2,
                             f"{self.value:.2f}")
            

class AngleWidget(QtWidgets.QWidget):
    """Display circle for drawing the exact polarization angle"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.theta = None
        self._background = None  # cached circle

        self.setMinimumSize(100, 100)

    def set_angle(self, theta):
        if self.theta == theta:
            return
        self.theta = theta
        self.update()

    def resizeEvent(self, event):
        self._generate_background()
        super().resizeEvent(event)

    def _generate_background(self):
        """Pre-render static circle + axes"""
        w, h = self.width(), self.height()
        if w <= 0 or h <= 0:
            return

        pixmap = QtGui.QPixmap(w, h)
        pixmap.fill(QtCore.Qt.GlobalColor.transparent)

        painter = QtGui.QPainter(pixmap)

        cx = w // 2
        cy = h // 2
        r = min(cx, cy) - 5

        fg = self.palette().color(QtGui.QPalette.ColorRole.WindowText)
        painter.setPen(QtGui.QPen(fg, 2))
        painter.drawEllipse(cx - r, cy - r, 2*r, 2*r)

        # axes
        painter.drawLine(cx - r, cy, cx + r, cy)
        painter.drawLine(cx, cy - r, cx, cy + r)

        painter.end()
        self._background = pixmap

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)

        # Draw cached background
        if self._background:
            painter.drawPixmap(0, 0, self._background)

        # Draw angle (dynamic)
        if self.theta is not None:
            rect = self.rect()
            cx = rect.width() // 2
            cy = rect.height() // 2
            r = min(cx, cy) - 5

            x1 = cx - r * np.cos(self.theta)
            y1 = cy + r * np.sin(self.theta)
            x2 = cx + r * np.cos(self.theta)
            y2 = cy - r * np.sin(self.theta)

            painter.setPen(QtGui.QPen(QtGui.QColor("red"), 5))
            painter.drawLine(int(x1), int(y1), int(x2), int(y2))
