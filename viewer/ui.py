import os
import sys

# Add __path__ to this module so submodules in viewer/ui/ can be imported
# as viewer.ui.xxx despite this file (viewer/ui.py) shadowing the package.
_ui_dir = os.path.join(os.path.dirname(__file__), "ui")
if not hasattr(sys.modules[__name__], "__path__"):
    sys.modules[__name__].__path__ = [_ui_dir]

from viewer.ui.render_settings import RenderSettings
from viewer.ui.panels import Panel, ScenePanel, RenderSettingsPanel
from viewer.ui.ui import UIManager, UI

__all__ = ["RenderSettings", "Panel", "ScenePanel", "RenderSettingsPanel", "UIManager", "UI"]
