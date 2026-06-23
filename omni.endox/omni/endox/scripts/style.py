"""
EndoX Pipeline - UI Styles
"""

import omni.ui as ui

LABEL_WIDTH = 170

main_style = {
    "Button": {
        "width": 0,
        "background_color": ui.color("#1a1a2e"),
    },
    "Button:hovered": {
        "background_color": ui.color("#16213e"),
    },
    "Button:pressed": {
        "background_color": ui.color("#0f3460"),
    },
    "HStack": {
        "padding": 2,
    },
}

SECTION_STYLE = {
    "margin": 4,
    "font_size": 14,
}

FIELD_STYLE = {
    "margin": 2,
}

TITLE_STYLE = {
    "font_size": 24,
    "margin": 8,
    "color": ui.color("#e0e0e0"),
}

SECTION_HEADER_STYLE = {
    "font_size": 18,
}

BUTTON_STYLE = {
    "font_size": 14,
}

