"""
EndoX Pipeline - Extension lifecycle
"""

import omni.ext

from .window import EndoxPipelineWindow


class EndoxPipelineExtension(omni.ext.IExt):
    """Omniverse extension entry-point for the EndoX Pipeline GUI."""

    WINDOW_NAME = "EndoX Pipeline"
    MENU_PATH = f"Window/{WINDOW_NAME}"

    def __init__(self) -> None:
        super().__init__()
        self._window: EndoxPipelineWindow | None = None
        self._menu = None

    # ── Lifecycle ────────────────────────────────────────────────────

    def on_startup(self, ext_id: str):
        print("[omni.endox] Extension startup")
        self._menu = omni.kit.ui.get_editor_menu().add_item(
            self.MENU_PATH, self._on_menu_click, toggle=True, value=True,
        )
        self._show_window(True)

    def on_shutdown(self):
        if self._menu:
            omni.kit.ui.get_editor_menu().remove_item(self.MENU_PATH)
            self._menu = None
        if self._window:
            self._window.destroy()
            self._window = None
        print("[omni.endox] Extension shutdown")

    # ── Window management ────────────────────────────────────────────

    def _on_menu_click(self, menu, value):
        self._show_window(value)

    def _show_window(self, visible: bool):
        omni.kit.ui.get_editor_menu().set_value(self.MENU_PATH, visible)
        if visible:
            self._window = EndoxPipelineWindow(
                self.WINDOW_NAME, width=520, height=900,
            )
            self._window.set_visibility_changed_fn(self._on_visibility_changed)
        elif self._window:
            self._window.visible = False

    def _on_visibility_changed(self, visible: bool):
        omni.kit.ui.get_editor_menu().set_value(self.MENU_PATH, visible)
        if not visible:
            self._window = None
