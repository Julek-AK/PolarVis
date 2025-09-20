from PyQt5.QtWidgets import QGraphicsView
from PyQt5.QtGui import QPainter

class GraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragMode(QGraphicsView.ScrollHandDrag)  # click + drag to pan
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)

        self._zoom = 0  # track zoom level

    def wheelEvent(self, event):
        """Zoom with mouse wheel"""
        zoom_in_factor = 1.25
        zoom_out_factor = 1 / zoom_in_factor

        # Save scene pos
        old_pos = self.mapToScene(event.pos())

        if event.angleDelta().y() > 0:
            zoom_factor = zoom_in_factor
            self._zoom += 1
        else:
            zoom_factor = zoom_out_factor
            self._zoom -= 1

        # Prevent zooming out too far or too close
        if self._zoom < -5:
            self._zoom = -5
            return
        
        if self._zoom > 15:
            self._zoom = 15
            return
        

        self.scale(zoom_factor, zoom_factor)

        # Get new position
        new_pos = self.mapToScene(event.pos())

        # Move scene to keep cursor position stable
        delta = new_pos - old_pos
        self.translate(delta.x(), delta.y())
