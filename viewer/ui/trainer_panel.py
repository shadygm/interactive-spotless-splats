import numpy as np
from imgui_bundle import imgui
from imgui_bundle import implot
from loguru import logger

from viewer.ui.panels import Panel


class TrainerPanel(Panel):
    """GUI panel for 3DGS training controls and loss visualization."""

    def __init__(self, trainer, scene_state):
        self.trainer = trainer
        self.scene_state = scene_state

        # Input buffers for settings
        self._max_splats = trainer.config.max_splats
        self._num_iterations = trainer.config.num_iterations
        self._strategy_names = ["mcmc", "default"]
        self._strategy_idx = self._strategy_names.index(trainer.config.strategy)

    def draw(self):
        has_colmap = self.scene_state.has_colmap

        imgui.text("Training Configuration")
        imgui.separator()

        # Max splats
        changed_max, new_max = imgui.input_int(
            "Max Splats", self._max_splats, step=1000, step_fast=10000
        )
        if changed_max:
            self._max_splats = max(1000, new_max)
            self.trainer.config.max_splats = self._max_splats

        # Num iterations
        changed_iter, new_iter = imgui.input_int(
            "Iterations", self._num_iterations, step=1000, step_fast=5000
        )
        if changed_iter:
            self._num_iterations = max(100, new_iter)
            self.trainer.config.num_iterations = self._num_iterations

        # Strategy selector
        changed_strat, self._strategy_idx = imgui.combo(
            "Strategy", self._strategy_idx, self._strategy_names
        )
        if changed_strat:
            self.trainer.config.strategy = self._strategy_names[self._strategy_idx]

        # SH degree selector
        sh_labels = ["0", "1", "2", "3"]
        sh_values = [0, 1, 2, 3]
        current_sh_idx = sh_values.index(self.trainer.config.sh_degree)
        changed_sh, new_sh_idx = imgui.combo("SH Degree", current_sh_idx, sh_labels)
        if changed_sh:
            self.trainer.config.sh_degree = sh_values[new_sh_idx]

        imgui.separator()

        # Status
        if self.trainer.is_running:
            imgui.text(self.trainer.status_message)
            imgui.text(f"Splats: {self.trainer.current_splats:,}")
        else:
            imgui.text(self.trainer.status_message)

        imgui.separator()

        # Start / Stop button
        if self.trainer.is_running:
            if imgui.button("Stop Training", imgui.ImVec2(imgui.get_content_region_avail().x, 0)):
                self.trainer.stop()
                logger.info("Training stop requested")
        else:
            imgui.begin_disabled(not has_colmap)
            if imgui.button("Start Training", imgui.ImVec2(imgui.get_content_region_avail().x, 0)):
                # Find the dataset path from scene_state
                dataset_path = getattr(self.scene_state, "_last_colmap_path", None)
                if dataset_path:
                    self.trainer.start(dataset_path)
                    logger.info(f"Started training from {dataset_path}")
                else:
                    logger.warning("No dataset path available for training")
            imgui.end_disabled()
            if not has_colmap:
                imgui.text_disabled("Load a dataset to enable training")

        imgui.separator()

        # Loss plot
        imgui.text("Loss History")
        with self.trainer._lock:
            loss_history = list(self.trainer.loss_history)

        if loss_history:
            iters = [x[0] for x in loss_history]
            losses = [x[1] for x in loss_history]

            plot_width = imgui.get_content_region_avail().x
            plot_height = 300

            # Tight padding: remove outer plot padding and border
            implot.push_style_var(implot.StyleVar_.plot_padding, imgui.ImVec2(0, 0))
            implot.push_style_var(implot.StyleVar_.plot_border_size, 0)
            if implot.begin_plot("Loss", size=(plot_width, plot_height)):
                implot.setup_axes("Iteration", "Loss")
                if iters:
                    implot.setup_axis_limits(
                        implot.ImAxis_.x1,
                        float(iters[0]),
                        float(iters[-1]),
                        implot.Cond_.always,
                    )
                    implot.setup_axis_limits(
                        implot.ImAxis_.y1,
                        min(losses) * 0.9,
                        max(losses) * 1.1,
                        implot.Cond_.always,
                    )
                implot.plot_line("Loss", np.array(iters, dtype=np.float32), np.array(losses, dtype=np.float32))
                implot.end_plot()
            implot.pop_style_var(2)
        else:
            imgui.text_disabled("No loss data yet. Start training to see the plot.")
