import Metashape
from PySide2 import QtWidgets

# Checking compatibility
compatible_major_version = "2.1"
found_major_version = ".".join(Metashape.app.version.split('.')[:2])
if found_major_version != compatible_major_version:
    raise Exception("Incompatible Metashape version: {} != {}".format(found_major_version, compatible_major_version))

class TiePointsRegionFilterDlg(QtWidgets.QDialog):

    def __init__(self, parent):
        QtWidgets.QDialog.__init__(self, parent)
        self.setWindowTitle("Tie Points Region Filter Tool")

        # Widgets
        self.applyFilterButton = QtWidgets.QPushButton("Filter Tie Points")
        self.closeButton = QtWidgets.QPushButton("Close")

        self.infoLabel = QtWidgets.QLabel("Ensure tie points are selected in the 3D view before running.")
        self.selectedPointsLabel = QtWidgets.QLabel(f"Selected Tie Points: {self.get_selected_tie_points_count()}")
        self.removeCamerasCheckbox = QtWidgets.QCheckBox("Remove Disabled Cameras")
        self.preserveRegionCheckbox = QtWidgets.QCheckBox("Preserve Tie Points Within Region")
        
        # Layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.infoLabel)
        layout.addWidget(self.selectedPointsLabel)
        layout.addWidget(self.preserveRegionCheckbox)
        layout.addWidget(self.removeCamerasCheckbox)
        layout.addWidget(self.applyFilterButton)
        layout.addWidget(self.closeButton)
        
        self.setLayout(layout)

        # Connect Actions
        self.applyFilterButton.clicked.connect(self.filter_tie_points_region)
        self.closeButton.clicked.connect(self.close)

    def get_selected_tie_points_count(self):
        """Return the number of selected tie points."""
        doc = Metashape.app.document
        chunk = doc.chunk

        if not chunk or not chunk.tie_points:
            return 0

        return sum(1 for point in chunk.tie_points.points if point.selected)

    def filter_tie_points_region(self):
        """Filter tie points based on their inclusion in the region."""
        doc = Metashape.app.document
        chunk = doc.chunk

        if not chunk or not chunk.tie_points:
            QtWidgets.QMessageBox.warning(self, "Error", "No active chunk or tie points found.")
            return

        # Get the selected tie points
        selected_points = [point for point in chunk.tie_points.points if point.selected]
        
        if not selected_points:
            QtWidgets.QMessageBox.warning(self, "Error", "No tie points selected.")
            return
        
        if not self.preserveRegionCheckbox.isChecked():
            # Remove all unselected tie points
            for point in chunk.tie_points.points:
                if not point.selected:
                    point.valid = False
                    
        # Region parameters
        region = chunk.region
        R = region.rot
        C = region.center
        size = region.size

        # Get selected point coordinates and calculate bounds in region space
        min_coord = Metashape.Vector([float('inf')] * 3)
        max_coord = Metashape.Vector([-float('inf')] * 3)

        for point in selected_points:
            coord = point.coord
            coord.size = 3
            v_c = coord - C
            v_r = R.t() * v_c

            min_coord = Metashape.Vector([min(min_coord[i], v_r[i]) for i in range(3)])
            max_coord = Metashape.Vector([max(max_coord[i], v_r[i]) for i in range(3)])

        # Update region
        new_center = (min_coord + max_coord) / 2.0
        new_size = max_coord - min_coord
        region.center = C + R * new_center
        region.size = new_size
        chunk.region = region
                
        # Filter tie points
        valid_points = []
        for point in chunk.tie_points.points:
            if not point.valid:
                continue

            coord = point.coord
            coord.size = 3
            v_c = coord - region.center
            v_r = R.t() * v_c

            if (
                -region.size.x / 2 <= v_r.x <= region.size.x / 2 and
                -region.size.y / 2 <= v_r.y <= region.size.y / 2 and
                -region.size.z / 2 <= v_r.z <= region.size.z / 2
            ):
                valid_points.append(point.track_id)
            else: 
                point.valid = False


        cameras_to_disable = []
        for camera in chunk.cameras:
            if not camera.enabled:
                continue

            projections = chunk.tie_points.projections[camera]
            if not any(proj.track_id in valid_points for proj in projections):
                cameras_to_disable.append(camera)

        for camera in cameras_to_disable:
            camera.enabled = False
            
        # Remove disabled cameras if checkbox is selected
        if self.removeCamerasCheckbox.isChecked():
            self.remove_disabled_cameras(chunk)

        QtWidgets.QMessageBox.information(self, "Filter Complete",
                                          f"Filtered {len(chunk.tie_points.points) - len(valid_points)} tie points.")

    def remove_disabled_cameras(self, chunk):
        """Removes all cameras in the chunk that are disabled."""
        chunk.remove([camera for camera in chunk.cameras if not camera.enabled])

def tie_points_region_filter_tool():
    """Launch the Tie Points Region Filter Tool."""
    app = QtWidgets.QApplication.instance()
    parent = app.activeWindow()

    dlg = TiePointsRegionFilterDlg(parent)
    dlg.exec_()


# Add script to Metashape menu
label = "Scripts/Tie Points Region Filter Tool"
Metashape.app.addMenuItem(label, tie_points_region_filter_tool)
print(f"To execute this script, select {label}")
