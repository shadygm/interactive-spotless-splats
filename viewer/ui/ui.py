from viewer.ui.panels import ScenePanel, RenderSettingsPanel
from viewer.ui.trainer_panel import TrainerPanel
from viewer.ui.spotless_panel import SpotlessPanel


class UIManager:
    def __init__(self, scene_state, renderer, render_settings, camera=None, trainer=None, on_go_to_frustum=None):
        self.scene_panel = ScenePanel(scene_state, renderer, render_settings)
        self.render_settings_panel = RenderSettingsPanel(render_settings, camera, scene_state)
        self.trainer_panel = TrainerPanel(trainer, scene_state, on_go_to_frustum=on_go_to_frustum) if trainer else None
        self.spotless_panel = SpotlessPanel(trainer, scene_state) if trainer else None
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

    def draw_trainer_panel(self):
        if self.trainer_panel:
            self.trainer_panel.draw()

    def draw_spotless_panel(self):
        if self.spotless_panel:
            self.spotless_panel.draw()

    @property
    def pending_dataset_path(self):
        return self.scene_panel.pending_dataset_path

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

    def __init__(self, scene_state, renderer, render_settings, camera=None, trainer=None, on_go_to_frustum=None):
        self._manager = UIManager(scene_state, renderer, render_settings, camera, trainer, on_go_to_frustum)
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

    def draw_trainer(self):
        return self._manager.draw_trainer_panel()

    def draw_spotless(self):
        return self._manager.draw_spotless_panel()

    @property
    def pending_dataset_path(self):
        return self._manager.pending_dataset_path

    @property
    def pending_ply_path(self):
        return self._manager.pending_ply_path

    def clear_pending_actions(self):
        return self._manager.clear_pending_actions()
