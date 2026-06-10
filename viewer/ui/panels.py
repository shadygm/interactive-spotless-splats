from abc import ABC, abstractmethod

from imgui_bundle import imgui
from imgui_bundle import portable_file_dialogs as pfd
from loguru import logger


class Panel(ABC):
    @abstractmethod
    def draw(self):
        ...


class ScenePanel(Panel):
    def __init__(self, scene_state, renderer, render_settings):
        self.scene_state = scene_state
        self.renderer = renderer
        self.render_settings = render_settings
        self._dataset_path_input = ""
        self._ply_path_input = ""
        self._dataset_dialog = None
        self._ply_dialog = None
        self.pending_dataset_path = None
        self.pending_ply_path = None

    def draw(self):
        if imgui.button("Import Dataset"):
            self._dataset_dialog = pfd.select_folder("Select dataset folder")
            logger.debug("Opened dataset folder dialog")

        if self._dataset_dialog is not None:
            if self._dataset_dialog.ready():
                result = self._dataset_dialog.result()
                self._dataset_dialog = None
                if result:
                    logger.info(f"Selected dataset folder: {result}")
                    self.pending_dataset_path = result

        if imgui.button("Import PLY Splat"):
            self._ply_dialog = pfd.open_file("Select PLY file", filters=["*.ply"])
            logger.debug("Opened PLY file dialog")

        if self._ply_dialog is not None:
            if self._ply_dialog.ready():
                result = self._ply_dialog.result()
                self._ply_dialog = None
                if result:
                    path = result[0] if isinstance(result, list) else result
                    logger.info(f"Selected PLY file: {path}")
                    self.pending_ply_path = path

        imgui.separator()

        num_images = len(self.scene_state.colmap_images) if self.scene_state.has_colmap else 0
        num_points = len(self.scene_state.colmap_points3D) if self.scene_state.has_colmap else 0
        num_gaussians = self.scene_state.gaussians["means"].shape[0] if self.scene_state.has_gaussians else 0
        fps = self.renderer.fps

        imgui.text("Stats")
        imgui.separator()
        imgui.text(f"Images:    {num_images:,}")
        imgui.text(f"Points:    {num_points:,}")
        imgui.text(f"Gaussians: {num_gaussians:,}")
        if fps > 0:
            imgui.text(f"FPS:       {fps:,.1f}")

    def clear_pending_actions(self):
        self.pending_dataset_path = None
        self.pending_ply_path = None


class RenderSettingsPanel(Panel):
    def __init__(self, render_settings, camera=None, scene_state=None):
        self.render_settings = render_settings
        self.camera = camera
        self.scene_state = scene_state

    def draw(self):
        if imgui.collapsing_header("Frustums", imgui.TreeNodeFlags_.default_open):
            changed_visible, new_visible = imgui.checkbox(
                "Show Camera Frustums", self.render_settings.show_frustums
            )
            if changed_visible:
                self.render_settings.show_frustums = new_visible
                logger.debug(f"Camera frustums visibility: {new_visible}")

            c = self.render_settings.frustum_color
            changed_color, new_color = imgui.color_edit3("Frustum color", c)
            if changed_color:
                self.render_settings.frustum_color = list(new_color)
                # Frustum color is a uniform, no VBO rebuild needed
                logger.debug("Frustum color changed (uniform only)")

            s = self.render_settings.frustum_size
            changed_size, new_size = imgui.slider_float(
                "Frustum size", s, 0.0001, 0.01, format="%.4f"
            )
            if changed_size:
                self.render_settings.frustum_size = new_size
                self.render_settings.bump_frustum()
                logger.debug("Frustum size changed")

        if imgui.collapsing_header("Points", imgui.TreeNodeFlags_.default_open):
            ps = self.render_settings.point_size
            changed_size, new_size = imgui.drag_float(
                "Point size", ps, 0.1, 1.0, 20.0, format="%.1f"
            )
            if changed_size:
                self.render_settings.point_size = new_size
                logger.debug("Point size changed")

        if imgui.collapsing_header("Splat", imgui.TreeNodeFlags_.default_open):
            # SH Degree
            sh_labels = ["0", "1", "2", "3"]
            sh_values = [0, 1, 2, 3]
            current_sh_idx = sh_values.index(self.render_settings.sh_degree)
            imgui.text("SH Degree")
            for i, label in enumerate(sh_labels):
                if i > 0:
                    imgui.same_line()
                if imgui.radio_button(label, current_sh_idx == i):
                    self.render_settings.sh_degree = sh_values[i]
                    logger.debug(f"SH degree set to {self.render_settings.sh_degree}")

            imgui.separator()

            # Render Mode
            current_mode = self.render_settings.render_mode
            imgui.text("Render Mode")
            if imgui.radio_button("RGB", current_mode == "RGB"):
                self.render_settings.render_mode = "RGB"
                logger.debug("Render mode set to RGB")
            imgui.same_line()
            if imgui.radio_button("Depth", current_mode == "Depth"):
                self.render_settings.render_mode = "Depth"
                logger.debug("Render mode set to Depth")

            if self.render_settings.render_mode == "Depth":
                imgui.separator()
                cmap_names = ["viridis", "plasma", "jet", "hot", "cool", "turbo", "grayscale"]
                current_idx = cmap_names.index(self.render_settings.depth_colormap)
                labels = ["B&W" if n == "grayscale" else n.capitalize() for n in cmap_names]
                changed, new_idx = imgui.combo("Depth Colormap", current_idx, labels)
                if changed:
                    self.render_settings.depth_colormap = cmap_names[new_idx]
                    logger.debug(f"Depth colormap set to {cmap_names[new_idx]}")

            imgui.separator()

            # Near/Far planes
            changed_near, new_near = imgui.slider_float("Near plane", self.render_settings.near_plane, 0.001, 10.0, format="%.3f")
            if changed_near:
                self.render_settings.near_plane = new_near
                logger.debug(f"Near plane changed to {new_near:.3f}")

            changed_far, new_far = imgui.slider_float("Far plane", self.render_settings.far_plane, 1.0, 5000.0, format="%.1f")
            if changed_far:
                self.render_settings.far_plane = new_far
                logger.debug(f"Far plane changed to {new_far:.1f}")

            imgui.separator()

            # Background color
            bg = self.render_settings.background_color
            changed_bg, new_bg = imgui.color_edit3("Background", bg)
            if changed_bg:
                self.render_settings.background_color = list(new_bg)
                logger.debug("Background color changed")

            imgui.separator()

            # Rasterize mode
            imgui.text("Rasterize Mode")
            if imgui.radio_button("Classic", self.render_settings.rasterize_mode == "classic"):
                self.render_settings.rasterize_mode = "classic"
                logger.debug("Rasterize mode set to classic")
            imgui.same_line()
            if imgui.radio_button("Antialiased", self.render_settings.rasterize_mode == "antialiased"):
                self.render_settings.rasterize_mode = "antialiased"
                logger.debug("Rasterize mode set to antialiased")

            imgui.separator()

            # eps2d
            changed_eps, new_eps = imgui.slider_float("Eps2D", self.render_settings.eps2d, 0.0, 1.0, format="%.2f")
            if changed_eps:
                self.render_settings.eps2d = new_eps
                logger.debug(f"Eps2D changed to {new_eps:.2f}")

            imgui.separator()

            # radius_clip
            changed_clip, new_clip = imgui.slider_float("Radius Clip", self.render_settings.radius_clip, 0.0, 5.0, format="%.1f")
            if changed_clip:
                self.render_settings.radius_clip = new_clip
                logger.debug(f"Radius clip changed to {new_clip:.1f}")

        if imgui.collapsing_header("Camera", imgui.TreeNodeFlags_.default_open):
            imgui.text("Control Mode")
            if imgui.radio_button("Orbit", self.render_settings.camera_mode == "orbit"):
                self.render_settings.camera_mode = "orbit"
                logger.debug("Camera mode set to orbit")
            imgui.same_line()
            if imgui.radio_button("FPS", self.render_settings.camera_mode == "fps"):
                self.render_settings.camera_mode = "fps"
                logger.debug("Camera mode set to fps")
            imgui.separator()

            if self.camera is not None:
                if imgui.button("Go to Home"):
                    self.camera.reset()
                    logger.info("Camera reset to home")

                imgui.same_line()
                if imgui.button("Fit to Scene"):
                    bmin, bmax = self.scene_state.get_scene_bounds()
                    self.camera.fit_to_bounds(bmin, bmax)
                    logger.info("Camera fitted to scene bounds")

                fov = self.camera.fov
                changed_fov, new_fov = imgui.slider_float("FOV", fov, 10.0, 120.0, format="%.1f")
                if changed_fov:
                    self.camera.fov = new_fov
                    logger.debug(f"Camera FOV changed to {new_fov:.1f}")

                if self.render_settings.camera_mode == "fps":
                    speed = getattr(self.camera, 'move_speed', 5.0)
                    changed_speed, new_speed = imgui.slider_float("Move Speed", speed, 1.0, 50.0, format="%.1f")
                    if changed_speed:
                        self.camera.move_speed = new_speed
                        logger.debug(f"Camera move speed changed to {new_speed:.1f}")

            else:
                imgui.text("Camera not available")
