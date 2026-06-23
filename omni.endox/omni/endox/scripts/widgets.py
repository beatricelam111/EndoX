"""
EndoX Pipeline - Reusable UI Widgets
"""

import omni.ui as ui
from omni.ui import color as cl

from .style import LABEL_WIDTH, FIELD_STYLE


class StatusLog:
    """Multi-line status / log area with timestamped entries."""

    def __init__(self, max_lines: int = 80):
        self._max_lines = max_lines
        self._lines: list[str] = []
        self._label: ui.Label | None = None
        self._build()

    def _build(self):
        with ui.VStack(height=0):
            self._label = ui.Label(
                "Ready.",
                word_wrap=True,
                style={"font_size": 13, "color": cl("#aaaaaa")},
                height=0,
            )

    def log(self, msg: str):
        import time
        ts = time.strftime("%H:%M:%S")
        self._lines.append(f"[{ts}] {msg}")
        if len(self._lines) > self._max_lines:
            self._lines = self._lines[-self._max_lines:]
        if self._label:
            self._label.text = "\n".join(self._lines)

    def clear(self):
        self._lines.clear()
        if self._label:
            self._label.text = "Ready."


class ProgressBar:
    """Animated indeterminate progress bar."""

    def __init__(self):
        self._bar = None
        self._left = None
        self._right = None
        self._build()

    def _build(self):
        with ui.VStack():
            self._bar = ui.HStack(height=6, visible=False)
            with self._bar:
                self._left = ui.Spacer(width=ui.Fraction(0.0))
                ui.Rectangle(width=60, style={"background_color": cl("#76b900")})
                self._right = ui.Spacer(width=ui.Fraction(1.0))

    async def play(self):
        import omni.kit.app
        app = omni.kit.app.get_app()
        fraction = 0.0
        while True:
            fraction = (fraction + 0.01) % 1.0
            self._left.width = ui.Fraction(fraction)
            self._right.width = ui.Fraction(1.0 - fraction)
            await app.next_update_async()

    def show(self, visible: bool = True):
        if self._bar:
            self._bar.visible = visible


def labeled_field(label, model, **kwargs):
    """Create a label + auto-typed input field row."""
    with ui.HStack(height=24, style=FIELD_STYLE):
        ui.Label(label, width=LABEL_WIDTH)
        if isinstance(model, ui.SimpleStringModel):
            return ui.StringField(model=model)
        elif isinstance(model, ui.SimpleFloatModel):
            return ui.FloatDrag(model=model, **kwargs)
        elif isinstance(model, ui.SimpleIntModel):
            return ui.IntDrag(model=model, **kwargs)
        elif isinstance(model, ui.SimpleBoolModel):
            cb = ui.CheckBox(model=model, width=20)
            ui.Spacer()
            return cb


def xyz_row(label, mx, my, mz, **kwargs):
    """Create a label + X / Y / Z float-drag row."""
    with ui.HStack(height=24, style=FIELD_STYLE):
        ui.Label(label, width=LABEL_WIDTH)
        ui.Label("X", width=14)
        ui.FloatDrag(model=mx, **kwargs)
        ui.Label("Y", width=14)
        ui.FloatDrag(model=my, **kwargs)
        ui.Label("Z", width=14)
        ui.FloatDrag(model=mz, **kwargs)
