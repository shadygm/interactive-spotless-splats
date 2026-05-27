from viewer.ui.panels import ScenePanel, RenderSettingsPanel


class UIManager:
    def __init__(self, scene_state, renderer, render_settings, camera=None):
        self.scene_panel = ScenePanel(scene_state, renderer, render_settings)
        self.render_settings_panel = RenderSettingsPanel(render_settings, camera, scene_state)
        self._camera = camera

    @property
    def camera(self):
        return self._camera

    @camera.setter
    def camera(self, value):
        self._camera = value
        self.render_settings_panel.camera = value

    def draw_scene_panel(self):
        self.scene_panel.draw()

    def draw_render_settings_panel(self):
        self.render_settings_panel.draw()

    @property
    def pending_colmap_path(self):
        return self.scene_panel.pending_colmap_path

    @property
    def pending_ply_path(self):
        return self.scene_panel.pending_ply_path

    def clear_pending_actions(self):
        self.scene_panel.clear_pending_actions()

    # Backward-compatible aliases
    def draw_content(self):
        return self.draw_scene_panel()

    def draw_render_settings(self):
        return self.draw_render_settings_panel()


class UI:
    """Backward-compatible UI wrapper that delegates to UIManager."""

    def __init__(self, scene_state, renderer, render_settings, camera=None):
        self._manager = UIManager(scene_state, renderer, render_settings, camera)
        self._camera = camera

    @property
    def camera(self):
        return self._camera

    @camera.setter
    def camera(self, value):
        self._camera = value
        self._manager.camera = value

    def draw_content(self):
        return self._manager.draw_scene_panel()

    def draw_render_settings(self):
        return self._manager.draw_render_settings_panel()

    @property
    def pending_colmap_path(self):
        return self._manager.pending_colmap_path

    @property
    def pending_ply_path(self):
        return self._manager.pending_ply_path

    def clear_pending_actions(self):
        return self._manager.clear_pending_actions()
