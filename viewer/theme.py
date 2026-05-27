from imgui_bundle import imgui


# Gruvbox Dark colors (normalized to 0-1)
_BG0 = imgui.ImVec4(0.168, 0.157, 0.133, 1.0)       # #282828
_BG1 = imgui.ImVec4(0.235, 0.219, 0.211, 1.0)       # #3c3836
_BG2 = imgui.ImVec4(0.314, 0.286, 0.271, 1.0)       # #504945
_BG3 = imgui.ImVec4(0.4, 0.361, 0.329, 1.0)         # #665c54
_FG0 = imgui.ImVec4(0.984, 0.945, 0.78, 1.0)        # #fbf1c7
_FG1 = imgui.ImVec4(0.922, 0.859, 0.698, 1.0)       # #ebdbb2
_FG2 = imgui.ImVec4(0.835, 0.769, 0.631, 1.0)       # #d5c4a1
_FG3 = imgui.ImVec4(0.733, 0.682, 0.576, 1.0)       # #bdae93
_FG4 = imgui.ImVec4(0.659, 0.6, 0.518, 1.0)         # #a89984
_RED = imgui.ImVec4(0.8, 0.141, 0.113, 1.0)          # #cc241d
_GREEN = imgui.ImVec4(0.596, 0.592, 0.102, 1.0)     # #98971a
_YELLOW = imgui.ImVec4(0.843, 0.6, 0.129, 1.0)       # #d79921
_BLUE = imgui.ImVec4(0.271, 0.522, 0.533, 1.0)      # #458588
_PURPLE = imgui.ImVec4(0.694, 0.384, 0.525, 1.0)   # #b16286
_AQUA = imgui.ImVec4(0.408, 0.616, 0.416, 1.0)      # #689d6a
_ORANGE = imgui.ImVec4(0.839, 0.365, 0.055, 1.0)    # #d65d0e


def apply_gruvbox_theme():
    """Apply Gruvbox Dark theme to ImGui."""
    style = imgui.get_style()

    # Backgrounds
    style.set_color_(imgui.Col_.window_bg, _BG0)
    style.set_color_(imgui.Col_.child_bg, _BG0)
    style.set_color_(imgui.Col_.popup_bg, _BG1)
    style.set_color_(imgui.Col_.border, _BG2)
    style.set_color_(imgui.Col_.border_shadow, imgui.ImVec4(0.0, 0.0, 0.0, 0.0))

    # Frames
    style.set_color_(imgui.Col_.frame_bg, _BG1)
    style.set_color_(imgui.Col_.frame_bg_hovered, _BG2)
    style.set_color_(imgui.Col_.frame_bg_active, _BG3)

    # Title
    style.set_color_(imgui.Col_.title_bg, _BG0)
    style.set_color_(imgui.Col_.title_bg_active, _BG1)
    style.set_color_(imgui.Col_.title_bg_collapsed, _BG0)
    style.set_color_(imgui.Col_.menu_bar_bg, _BG0)

    # Scrollbar
    style.set_color_(imgui.Col_.scrollbar_bg, _BG1)
    style.set_color_(imgui.Col_.scrollbar_grab, _BG3)
    style.set_color_(imgui.Col_.scrollbar_grab_hovered, _FG4)
    style.set_color_(imgui.Col_.scrollbar_grab_active, _FG3)

    # Controls
    style.set_color_(imgui.Col_.check_mark, _GREEN)
    style.set_color_(imgui.Col_.slider_grab, _AQUA)
    style.set_color_(imgui.Col_.slider_grab_active, _BLUE)
    style.set_color_(imgui.Col_.button, _BG1)
    style.set_color_(imgui.Col_.button_hovered, _BG2)
    style.set_color_(imgui.Col_.button_active, _BG3)
    style.set_color_(imgui.Col_.header, _BG1)
    style.set_color_(imgui.Col_.header_hovered, _BG2)
    style.set_color_(imgui.Col_.header_active, _BG3)

    # Separators
    style.set_color_(imgui.Col_.separator, _BG2)
    style.set_color_(imgui.Col_.separator_hovered, _ORANGE)
    style.set_color_(imgui.Col_.separator_active, _ORANGE)

    # Resize grip
    style.set_color_(imgui.Col_.resize_grip, _BG3)
    style.set_color_(imgui.Col_.resize_grip_hovered, _ORANGE)
    style.set_color_(imgui.Col_.resize_grip_active, _ORANGE)

    # Tabs
    style.set_color_(imgui.Col_.tab, _BG1)
    style.set_color_(imgui.Col_.tab_hovered, _BG3)
    style.set_color_(imgui.Col_.tab_selected, _BG2)
    style.set_color_(imgui.Col_.tab_dimmed, _BG1)
    style.set_color_(imgui.Col_.tab_dimmed_selected, _BG2)

    # Docking
    style.set_color_(imgui.Col_.docking_preview, _AQUA)
    style.set_color_(imgui.Col_.docking_empty_bg, _BG0)

    # Plotting
    style.set_color_(imgui.Col_.plot_lines, _BLUE)
    style.set_color_(imgui.Col_.plot_lines_hovered, _ORANGE)
    style.set_color_(imgui.Col_.plot_histogram, _AQUA)
    style.set_color_(imgui.Col_.plot_histogram_hovered, _GREEN)

    # Tables
    style.set_color_(imgui.Col_.table_header_bg, _BG1)
    style.set_color_(imgui.Col_.table_border_strong, _BG2)
    style.set_color_(imgui.Col_.table_border_light, _BG1)
    style.set_color_(imgui.Col_.table_row_bg, imgui.ImVec4(0.0, 0.0, 0.0, 0.0))
    style.set_color_(imgui.Col_.table_row_bg_alt, imgui.ImVec4(1.0, 1.0, 1.0, 0.03))

    # Misc
    style.set_color_(imgui.Col_.text_selected_bg, _BG3)
    style.set_color_(imgui.Col_.drag_drop_target, _YELLOW)
    style.set_color_(imgui.Col_.nav_cursor, _PURPLE)
    style.set_color_(imgui.Col_.nav_windowing_highlight, imgui.ImVec4(1.0, 1.0, 1.0, 0.7))
    style.set_color_(imgui.Col_.nav_windowing_dim_bg, imgui.ImVec4(0.8, 0.8, 0.8, 0.2))
    style.set_color_(imgui.Col_.modal_window_dim_bg, imgui.ImVec4(0.0, 0.0, 0.0, 0.35))

    # Text
    style.set_color_(imgui.Col_.text, _FG0)
    style.set_color_(imgui.Col_.text_disabled, _FG4)

    # Rounded corners
    style.window_rounding = 6.0
    style.child_rounding = 6.0
    style.frame_rounding = 4.0
    style.popup_rounding = 4.0
    style.scrollbar_rounding = 9.0
    style.grab_rounding = 4.0
    style.tab_rounding = 4.0

    # Padding and spacing
    style.window_padding = imgui.ImVec2(8.0, 8.0)
    style.frame_padding = imgui.ImVec2(5.0, 3.0)
    style.item_spacing = imgui.ImVec2(8.0, 4.0)
    style.item_inner_spacing = imgui.ImVec2(4.0, 4.0)
    style.indent_spacing = 21.0
    style.scrollbar_size = 14.0
    style.grab_min_size = 10.0

    # Borders
    style.window_border_size = 1.0
    style.child_border_size = 1.0
    style.popup_border_size = 1.0
    style.frame_border_size = 0.0
    style.tab_border_size = 0.0
