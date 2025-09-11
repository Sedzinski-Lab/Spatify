import os
import itertools
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import napari
import tifffile
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton,
    QSpinBox, QSlider, QColorDialog, QListWidget, QListWidgetItem
)
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor

'''
pip install "napari[all]" magicgui tifffile pandas numpy
'''
class SpatialTranscriptomicsWidget(QWidget):
    def __init__(self, viewer: napari.Viewer, data_dir: str):
        super().__init__()
        self.viewer = viewer
        self.base_dir = Path(data_dir)
        self.current_data: Optional[pd.DataFrame] = None
        self.current_dapi_layer = None

        # per-layer storage: name -> napari layer
        self.gene_layers: Dict[str, napari.layers.Points] = {}

        # color cycle for new layers (distinct defaults)
        self.color_cycle = itertools.cycle([
            "red", "lime", "cyan", "magenta", "yellow", "orange",
            "blue", "white", "hotpink", "aqua", "chartreuse", "gold",
            "purple", "deepskyblue", "springgreen", "goldenrod"
        ])

        # Discover stages + current gene list
        self.time_stages = self._discover_time_stages()
        self.current_genes: List[str] = []

        # UI + state
        self._suspend_property_sync = False  # prevent feedback loops
        self.setup_ui()

    # ------------------ discovery & data loading ------------------
    def _discover_time_stages(self) -> List[str]:
        stages = []
        for item in self.base_dir.iterdir():
            if item.is_dir() and any(c.isdigit() for c in item.name):
                stages.append(item.name)
        # sort like 8, 10_5, 12 → (replace '_' with '.') safely
        def keyfun(s: str):
            try:
                return float(s.replace('_', '.'))
            except ValueError:
                return s
        return sorted(stages, key=keyfun)

    def load_stage_data(self, stage: str):
        stage_dir = self.base_dir / stage

        # DAPI
        dapi_files = list(stage_dir.glob("*DAPI.tiff")) or list(stage_dir.glob("*DAPI.tif"))
        if not dapi_files:
            raise FileNotFoundError(f"No *DAPI.tif(f) found in {stage_dir}")
        dapi_path = dapi_files[0]
        dapi_image = tifffile.imread(dapi_path)

        # transcripts
        txt_files = list(stage_dir.glob("*.txt"))
        if not txt_files:
            raise FileNotFoundError(f"No .txt transcript table found in {stage_dir}")
        txt_path = txt_files[0]
        df = pd.read_csv(txt_path, sep="\t", header=None)
        df.columns = ['x', 'y', 'z', 'gene', 'score']

        self.current_data = df
        self.current_genes = sorted(df['gene'].unique())
        return dapi_image, df

    # ------------------ UI ------------------
    def setup_ui(self):
        layout = QVBoxLayout(self)

        # Stage selection
        layout.addWidget(QLabel("Time Stage:"))
        self.stage_combo = QComboBox()
        self.stage_combo.addItems(self.time_stages)
        self.stage_combo.currentTextChanged.connect(self.load_stage)
        layout.addWidget(self.stage_combo)

        # Load btn (optional, also auto on change)
        load_btn = QPushButton("Load Stage")
        load_btn.clicked.connect(self.load_stage)
        layout.addWidget(load_btn)

        layout.addWidget(QLabel(""))  # spacer

        # Gene selection + Add
        layout.addWidget(QLabel("Gene:"))
        self.gene_combo = QComboBox()
        layout.addWidget(self.gene_combo)

        add_btn = QPushButton("Add Gene Layer")
        add_btn.clicked.connect(self.add_gene_layer)
        layout.addWidget(add_btn)

        layout.addWidget(QLabel(""))  # spacer

        # Gene layers list
        layout.addWidget(QLabel("Gene layers (select one to edit):"))
        self.layers_list = QListWidget()
        self.layers_list.currentItemChanged.connect(self._on_layer_selection_changed)
        self.layers_list.itemDoubleClicked.connect(self._focus_selected_layer)
        layout.addWidget(self.layers_list)

        # Visualization controls (apply to selected layer only)
        layout.addWidget(QLabel("Layer properties:"))

        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("Point size:"))
        self.size_spin = QSpinBox()
        self.size_spin.setRange(1, 50)
        self.size_spin.setValue(5)
        self.size_spin.valueChanged.connect(self._apply_props_to_selected_layer)
        size_w = QWidget(); size_w.setLayout(size_row); size_row.addWidget(self.size_spin)
        layout.addWidget(size_w)

        color_row = QHBoxLayout()
        color_row.addWidget(QLabel("Color:"))
        self.color_btn = QPushButton("Pick…")
        self.color_btn.clicked.connect(self._pick_color_for_selected_layer)
        color_row.addWidget(self.color_btn)
        color_w = QWidget(); color_w.setLayout(color_row)
        layout.addWidget(color_w)

        op_row = QHBoxLayout()
        op_row.addWidget(QLabel("Opacity:"))
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(80)
        self.opacity_slider.valueChanged.connect(self._apply_props_to_selected_layer)
        op_row.addWidget(self.opacity_slider)
        op_w = QWidget(); op_w.setLayout(op_row)
        layout.addWidget(op_w)

        # Clear gene layers
        clear_btn = QPushButton("Clear Gene Layers")
        clear_btn.clicked.connect(self.clear_gene_layers)
        layout.addWidget(clear_btn)

        # init with first stage
        self.load_stage()

    # ------------------ stage load ------------------
    def load_stage(self):
        selected_stage = self.stage_combo.currentText()
        if not selected_stage:
            return
        try:
            dapi_image, transcript_df = self.load_stage_data(selected_stage)

            # clear all layers (fresh stage)
            self.viewer.layers.clear()
            self.gene_layers.clear()
            self.layers_list.clear()

            # Add DAPI (grayscale with auto contrast)
            self.current_dapi_layer = self.viewer.add_image(
                dapi_image,
                name=f"DAPI_{selected_stage}",
                colormap="gray",
                contrast_limits=[float(np.min(dapi_image)), float(np.max(dapi_image))],
                blending="translucent"
            )

            # update gene options
            self.ggene_block(True)
            self.gene_combo.clear()
            self.gene_combo.addItems(self.current_genes)
            self.ggene_block(False)

            # reset color cycle for new stage
            self.color_cycle = itertools.cycle([
                "red", "lime", "cyan", "magenta", "yellow", "orange",
                "blue", "white", "hotpink", "aqua", "chartreuse", "gold",
                "purple", "deepskyblue", "springgreen", "goldenrod"
            ])

            print(f"Loaded stage {selected_stage}: {len(transcript_df)} rows, {len(self.current_genes)} genes")

        except Exception as e:
            print(f"Error loading stage {selected_stage}: {e}")

    def ggene_block(self, block: bool):
        self.gene_combo.blockSignals(block)

    # ------------------ layers ------------------
    def add_gene_layer(self):
        """Add (or replace) a points layer for the selected gene; assigns its own color/size/opacity."""
        if self.current_data is None:
            return
        gene = self.gene_combo.currentText()
        if not gene:
            return

        df_gene = self.current_data[self.current_data['gene'] == gene]
        if df_gene.empty:
            print(f"No data for gene {gene}")
            return

        # 2D points (y, x). If you want 3D, pass (z, y, x) and set ndim correctly.
        points = df_gene[['y', 'x']].values

        # unique layer name for this stage+gene
        stage = self.stage_combo.currentText()
        layer_name = f"{stage}:{gene}"

        # pick default props (independent for each layer)
        default_size = max(3, self.size_spin.value())  # use current control as default for new
        default_opacity = self.opacity_slider.value() / 100.0
        default_color = next(self.color_cycle)

        # replace existing layer if exists
        if layer_name in self.gene_layers and self.gene_layers[layer_name] in self.viewer.layers:
            lyr = self.gene_layers[layer_name]
            lyr.data = points
            # keep that layer’s existing look
        else:
            lyr = self.viewer.add_points(
                points,
                name=layer_name,
                size=default_size,
                face_color=default_color,
                edge_color=default_color,
                opacity=default_opacity,
                blending="additive"
            )
            self.gene_layers[layer_name] = lyr
            # add to list UI
            item = QListWidgetItem(layer_name)
            self.layers_list.addItem(item)
            self.layers_list.setCurrentItem(item)  # select it so controls show its props

        print(f"Added/updated layer {layer_name}: {len(points)} points")

    def _on_layer_selection_changed(self, curr: Optional[QListWidgetItem], prev: Optional[QListWidgetItem]):
        """When user selects a layer in the list, sync controls to *that layer's* properties."""
        if curr is None:
            return
        name = curr.text()
        lyr = self.gene_layers.get(name)
        if lyr is None or lyr not in self.viewer.layers:
            return
        try:
            self._suspend_property_sync = True
            # read props into controls
            # napari Points.size can be scalar or array; if array, take median
            size_val = int(np.median(lyr.size)) if np.ndim(lyr.size) else int(lyr.size)
            self.size_spin.setValue(max(1, size_val))
            self.opacity_slider.setValue(int(round(float(lyr.opacity) * 100)))

            # face_color can be name or RGBA; set button color visually
            color = lyr.face_color
            # convert to hex if array
            if isinstance(color, str):
                hexcolor = QColor(color).name()
            else:
                arr = np.array(color)
                if arr.ndim > 1:
                    arr = arr[0]
                r, g, b = (arr[:3] * 255).astype(int)
                hexcolor = QColor(r, g, b).name()
            self._set_color_button(hexcolor)
        finally:
            self._suspend_property_sync = False

    def _focus_selected_layer(self, item: QListWidgetItem):
        """Double-click a layer to select/focus it in napari."""
        name = item.text()
        if name in self.viewer.layers:
            self.viewer.layers.selection = [self.viewer.layers[name]]

    def _apply_props_to_selected_layer(self):
        """Apply controls to the currently selected layer only."""
        if self._suspend_property_sync:
            return
        item = self.layers_list.currentItem()
        if item is None:
            return
        name = item.text()
        lyr = self.gene_layers.get(name)
        if lyr is None or lyr not in self.viewer.layers:
            return

        size = self.size_spin.value()
        opacity = self.opacity_slider.value() / 100.0
        # keep color as is (only changed via _pick_color_for_selected_layer)
        lyr.size = size
        lyr.opacity = opacity

    def _pick_color_for_selected_layer(self):
        item = self.layers_list.currentItem()
        if item is None:
            return
        name = item.text()
        lyr = self.gene_layers.get(name)
        if lyr is None or lyr not in self.viewer.layers:
            return

        # current color as start
        start = QColor("#ff0000")
        fc = lyr.face_color
        if isinstance(fc, str):
            start = QColor(fc)
        else:
            arr = np.array(fc)
            if arr.ndim > 1:
                arr = arr[0]
            r, g, b = (arr[:3] * 255).astype(int)
            start = QColor(r, g, b)

        color = QColorDialog.getColor(start, self, f"Pick color for {name}")
        if color.isValid():
            hexc = color.name()
            lyr.face_color = hexc
            lyr.edge_color = hexc
            self._set_color_button(hexc)

    def _set_color_button(self, hexc: str):
        self.color_btn.setText(hexc.upper())
        # choose readable text color
        q = QColor(hexc)
        luminance = 0.299*q.red() + 0.587*q.green() + 0.114*q.blue()
        fg = "black" if luminance > 186 else "white"
        self.color_btn.setStyleSheet(f"background-color: {hexc}; color: {fg};")

    def clear_gene_layers(self):
        for name, lyr in list(self.gene_layers.items()):
            if lyr in self.viewer.layers:
                self.viewer.layers.remove(lyr)
        self.gene_layers.clear()
        self.layers_list.clear()
        print("Cleared all gene layers")


def create_st_viewer(data_dir: str = "Resolve files"):
    viewer = napari.Viewer()
    widget = SpatialTranscriptomicsWidget(viewer, data_dir)
    viewer.window.add_dock_widget(widget, area="right", name="Spatial Transcriptomics")
    return viewer, widget


if __name__ == "__main__":
    # Edit this path if needed
    viewer, widget = create_st_viewer("D:/Ziwei/Github/STdata/Resolve files")
    napari.run()
