from PyQt6.QtWidgets import QGraphicsView, QFileDialog, QGraphicsPixmapItem
from PyQt6.QtGui import QPainter, QPixmap
from PyQt6.QtCore import Qt

class ImageView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Enable panning (dragging)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)

        # Enable high-quality rendering
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing |
            QPainter.RenderHint.SmoothPixmapTransform
        )

        self._zoom = 0  # Track zoom level

    def wheelEvent(self, event):
        """Zoom with mouse wheel"""
        zoom_in_factor = 1.25
        zoom_out_factor = 1 / zoom_in_factor

        old_pos = self.mapToScene(event.position().toPoint())

        if event.angleDelta().y() > 0:
            zoom_factor = zoom_in_factor
            self._zoom += 1
        else:
            zoom_factor = zoom_out_factor
            self._zoom -= 1

        # Prevent zooming too far
        if self._zoom < -5:
            self._zoom = -5
            return
        if self._zoom > 15:
            self._zoom = 15
            return

        self.scale(zoom_factor, zoom_factor)

        new_pos = self.mapToScene(event.position().toPoint())
        delta = new_pos - old_pos
        self.translate(delta.x(), delta.y())



def load_image(window):
    # Open file dialog
    file_name, _ = QFileDialog.getOpenFileName(
        window,
        "Open Image",
        "",
        "Images (*.png *.jpg *.jpeg *.bmp *.gif);;All Files (*)"
    )

    if file_name:
        print(f"Opened the file: {file_name}")
        pixmap = QPixmap(file_name)

        # Clear any previous items
        window.scene.clear()

        # Add new image
        item = QGraphicsPixmapItem(pixmap)
        window.scene.addItem(item)
