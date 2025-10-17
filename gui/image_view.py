
# Builtins
from PIL import Image

# External Imports
from PyQt6.QtWidgets import QGraphicsView, QGraphicsPixmapItem, QGraphicsScene
from PyQt6.QtGui import QPainter, QPixmap, QImage
from PyQt6 import QtCore

# Internal Support



class ImageView(QGraphicsView):
    pixelHovered = QtCore.pyqtSignal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)

        # Enable panning (dragging)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)

        # Enable high-quality rendering
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing |
            QPainter.RenderHint.TextAntialiasing
        )

        self._zoom = 0  # Track zoom level

        # Mouse position tracking
        self.setMouseTracking(True)
        self._pixmap_item = None 

    def wheelEvent(self, event):
        """Zoom with mouse wheel"""
        zoom_in_factor: float = 1.25
        zoom_out_factor: float = 1 / zoom_in_factor

        old_pos = self.mapToScene(event.position().toPoint())

        if event.angleDelta().y() > 0:
            zoom_factor = zoom_in_factor
            self._zoom += 1
        else:
            zoom_factor = zoom_out_factor
            self._zoom -= 1

        # Prevent zooming too far out or in
        if self._zoom < -5:
            self._zoom = -5
            return
        if self._zoom > 20:
            self._zoom = 20
            return

        self.scale(zoom_factor, zoom_factor)

        new_pos = self.mapToScene(event.position().toPoint())
        delta = new_pos - old_pos
        self.translate(delta.x(), delta.y())

    def mouseMoveEvent(self, event):
        """Keeps track of which pixel the mouse is pointing at"""
        if self._pixmap_item is None:
            return

        scene_pos = self.mapToScene(event.pos())
        item_pos = self._pixmap_item.mapFromScene(scene_pos)
        img_x = int(item_pos.x())
        img_y = int(item_pos.y())

        if 0 <= img_x < self._pixmap_item.pixmap().width() and 0 <= img_y < self._pixmap_item.pixmap().height():
            self.pixelHovered.emit(img_x, img_y)

        super().mouseMoveEvent(event)
    
    # =============================================
    # IMAGE DISPLAYING
    # =============================================
    def display_image(self, window, file_name: str) -> None:
        """Display image from a file path"""
        file_name = str(file_name)
        print(f"[ImageView] Loading: {file_name}")

        pixmap = QPixmap(file_name)
        if pixmap.isNull():
            raise ValueError(f"[ImageView] Failed to load image: {file_name}")

        self._show_pixmap(window, pixmap)

    def display_pil_image(self, window, pil_image: Image.Image) -> None:
        """Display an image from a PIL.Image (used for generated viusalisation)"""
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        data = pil_image.tobytes("raw", "RGB")
        qimage = QImage(
            data,
            pil_image.width,
            pil_image.height,
            QImage.Format.Format_RGB888
        )
        
        pixmap = QPixmap.fromImage(qimage)
        self._pixmap_item = QGraphicsPixmapItem(pixmap)
        self._show_pixmap(window, pixmap)

    def _show_pixmap(self, window, pixmap: QPixmap) -> None:
        """Internal helper to clear the scene and show a pixmap"""
        if not hasattr(window, "scene") or window.scene is None:
            raise ValueError("[ImageView] No QGraphicsScene assigned to window")

        window.scene.clear()
        item = QGraphicsPixmapItem(pixmap)
        window.scene.addItem(item)
        # self._zoom = 0
        # self.resetTransform()