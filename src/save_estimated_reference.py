# Saves estimated values from the Reference pane to file (for GPS/INS sensor).
#
# This is python script for Metashape Pro. Scripts repository: https://github.com/agisoft-llc/metashape-scripts

import Metashape
import math

# Checking compatibility
compatible_major_version = "2.0"
found_major_version = ".".join(Metashape.app.version.split('.')[:2])
if found_major_version != compatible_major_version:
    raise Exception("Incompatible Metashape version: {} != {}".format(found_major_version, compatible_major_version))


class CameraStats():
    def __init__(self, camera):
        chunk = camera.chunk

        self.camera = camera
        self.estimated_location = None
        self.estimated_rotation = None
        self.reference_location = None
        self.reference_rotation = None
        self.error_location = None
        self.error_rotation = None
        self.sigma_location = None
        self.sigma_rotation = None

        if not camera.transform:
            return

        transform = chunk.transform.matrix
        crs = chunk.crs

        if chunk.camera_crs:
            transform = Metashape.CoordinateSystem.datumTransform(crs, chunk.camera_crs) * transform
            crs = chunk.camera_crs

        ecef_crs = self.getCartesianCrs(crs)

        camera_transform = transform * camera.transform
        antenna_transform = self.getAntennaTransform(camera.sensor)
        location_ecef = camera_transform.translation() + camera_transform.rotation() * antenna_transform.translation()
        rotation_ecef = camera_transform.rotation() * antenna_transform.rotation()

        self.estimated_location = Metashape.CoordinateSystem.transform(location_ecef, ecef_crs, crs)
        if camera.reference.location:
            self.reference_location = camera.reference.location
            self.error_location = Metashape.CoordinateSystem.transform(self.estimated_location, crs, ecef_crs) - Metashape.CoordinateSystem.transform(self.reference_location, crs, ecef_crs)
            self.error_location = crs.localframe(location_ecef).rotation() * self.error_location

        if chunk.euler_angles == Metashape.EulerAnglesOPK or chunk.euler_angles == Metashape.EulerAnglesPOK:
            localframe = crs.localframe(location_ecef)
        else:
            localframe = ecef_crs.localframe(location_ecef)

        self.estimated_rotation = Metashape.utils.mat2euler(localframe.rotation() * rotation_ecef, chunk.euler_angles)
        if camera.reference.rotation:
            self.reference_rotation = camera.reference.rotation
            self.error_rotation = self.estimated_rotation - self.reference_rotation
            self.error_rotation.x = (self.error_rotation.x + 180) % 360 - 180
            self.error_rotation.y = (self.error_rotation.y + 180) % 360 - 180
            self.error_rotation.z = (self.error_rotation.z + 180) % 360 - 180

        if camera.location_covariance:
            T = crs.localframe(location_ecef) * transform
            R = T.rotation() * T.scale()

            cov = R * camera.location_covariance * R.t()
            self.sigma_location = Metashape.Vector([math.sqrt(cov[0, 0]), math.sqrt(cov[1, 1]), math.sqrt(cov[2, 2])])

        if camera.rotation_covariance:
            T = localframe * camera_transform  # to reflect rotation angles ypr (ecef_crs.localfram) or opk (crs.localframe)
            R0 = T.rotation()

            dR = antenna_transform.rotation()

            da = Metashape.utils.dmat2euler(R0 * dR, R0 * self.makeRotationDx(0) * dR, chunk.euler_angles);
            db = Metashape.utils.dmat2euler(R0 * dR, R0 * self.makeRotationDy(0) * dR, chunk.euler_angles);
            dc = Metashape.utils.dmat2euler(R0 * dR, R0 * self.makeRotationDz(0) * dR, chunk.euler_angles);

            R = Metashape.Matrix([da, db, dc]).t()

            cov = R * camera.rotation_covariance * R.t()

            self.sigma_rotation = Metashape.Vector([math.sqrt(cov[0, 0]), math.sqrt(cov[1, 1]), math.sqrt(cov[2, 2])])

    def getCartesianCrs(self, crs):
        ecef_crs = crs.geoccs
        if ecef_crs is None:
            ecef_crs = Metashape.CoordinateSystem('LOCAL')
        return ecef_crs

    def getAntennaTransform(self, sensor):
        location = sensor.antenna.location
        if location is None:
            location = sensor.antenna.location_ref
        if location is None:
            location = Metashape.Vector([0.0, 0.0, 0.0])
        rotation = sensor.antenna.rotation
        if rotation is None:
            rotation = sensor.antenna.rotation_ref
        if rotation is None:
            rotation = Metashape.Vector([0.0, 0.0, 0.0])
        return Metashape.Matrix.Diag((1, -1, -1, 1)) * Metashape.Matrix.Translation(location) * Metashape.Matrix.Rotation(Metashape.Utils.ypr2mat(rotation))

    def makeRotationDx(self, alpha):
        sina = math.sin(alpha)
        cosa = math.cos(alpha)
        return Metashape.Matrix([[0, 0, 0], [0, -sina, -cosa], [0, cosa, -sina]])

    def makeRotationDy(self, alpha):
        sina = math.sin(alpha)
        cosa = math.cos(alpha)
        return Metashape.Matrix([[-sina, 0, cosa], [0, 0, 0], [-cosa, 0, -sina]])

    def makeRotationDz(self, alpha):
        sina = math.sin(alpha)
        cosa = math.cos(alpha)
        return Metashape.Matrix([[-sina, -cosa, 0], [cosa, -sina, 0], [0, 0, 0]])

    def getEulerAnglesName(self, euler_angles):
        if euler_angles == Metashape.EulerAnglesOPK:
            return "OPK"
        if euler_angles == Metashape.EulerAnglesPOK:
            return "POK"
        if euler_angles == Metashape.EulerAnglesYPR:
            return "YPR"
        if euler_angles == Metashape.EulerAnglesANK:
            return "ANK"

    def printVector(self, f, name, value, precision):
        fmt = "{:." + str(precision) + "f}"
        fmt = "    " + name + ": " + fmt + " " + fmt + " " + fmt + "\n"
        f.write(fmt.format(value.x, value.y, value.z))

    def write(self, f):
        euler_name = self.getEulerAnglesName(self.camera.chunk.euler_angles)

        f.write(self.camera.label + "\n")
        if self.reference_location:
            self.printVector(f, "   XYZ source", self.reference_location, 6)
        if self.error_location:
            self.printVector(f, "   XYZ error", self.error_location, 6)
        if self.estimated_location:
            self.printVector(f, "   XYZ estimated", self.estimated_location, 6)
        if self.sigma_location:
            self.printVector(f, "   XYZ sigma", self.sigma_location, 6)
        if self.reference_rotation:
            self.printVector(f, "   " + euler_name + " source", self.reference_rotation, 3)
        if self.error_rotation:
            self.printVector(f, "   " + euler_name + " error", self.error_rotation, 3)
        if self.estimated_rotation:
            self.printVector(f, "   " + euler_name + " estimated", self.estimated_rotation, 3)
        if self.sigma_rotation:
            self.printVector(f, "   " + euler_name + " sigma", self.sigma_rotation, 3)


class MarkerStats():
    def __init__(self, marker):
        chunk = marker.chunk

        self.marker = marker
        self.estimated_location = None
        self.reference_location = None
        self.error_location = None
        self.sigma_location = None

        if not marker.position:
            return

        transform = chunk.transform.matrix
        crs = chunk.crs

        if chunk.marker_crs:
            transform = Metashape.CoordinateSystem.datumTransform(crs, chunk.marker_crs) * transform
            crs = chunk.marker_crs

        ecef_crs = self.getCartesianCrs(crs)

        location_ecef = transform.mulp(marker.position)

        self.estimated_location = Metashape.CoordinateSystem.transform(location_ecef, ecef_crs, crs)
        if marker.reference.location:
            self.reference_location = marker.reference.location
            self.error_location = Metashape.CoordinateSystem.transform(self.estimated_location, crs, ecef_crs) - Metashape.CoordinateSystem.transform(self.reference_location, crs, ecef_crs)
            self.error_location = crs.localframe(location_ecef).rotation() * self.error_location

        if marker.position_covariance:
            T = crs.localframe(location_ecef) * transform
            R = T.rotation() * T.scale()

            cov = R * marker.position_covariance * R.t()
            self.sigma_location = Metashape.Vector([math.sqrt(cov[0, 0]), math.sqrt(cov[1, 1]), math.sqrt(cov[2, 2])])

    def getCartesianCrs(self, crs):
        ecef_crs = crs.geoccs
        if ecef_crs is None:
            ecef_crs = Metashape.CoordinateSystem('LOCAL')
        return ecef_crs

    def printVector(self, f, name, value, precision):
        fmt = "{:." + str(precision) + "f}"
        fmt = "    " + name + ": " + fmt + " " + fmt + " " + fmt + "\n"
        f.write(fmt.format(value.x, value.y, value.z))

    def write(self, f):
        f.write(self.marker.label + "\n")
        if self.reference_location:
            self.printVector(f, "   XYZ source", self.reference_location, 6)
        if self.error_location:
            self.printVector(f, "   XYZ error", self.error_location, 6)
        if self.estimated_location:
            self.printVector(f, "   XYZ estimated", self.estimated_location, 6)
        if self.sigma_location:
            self.printVector(f, "   XYZ sigma", self.sigma_location, 6)


class ScalebarStats():
    def __init__(self, scalebar):
        chunk = scalebar.chunk

        self.scalebar = scalebar
        self.estimated_distance = None
        self.reference_distance = None
        self.error_distance = None

        if not chunk.transform.scale:
            return

        point0 = None
        point1 = None

        if type(scalebar.point0) == Metashape.Marker:
            point0 = scalebar.point0.position
        if type(scalebar.point0) == Metashape.Camera:
            point0 = scalebar.point0.center
        if type(scalebar.point1) == Metashape.Marker:
            point1 = scalebar.point1.position
        if type(scalebar.point1) == Metashape.Camera:
            point1 = scalebar.point1.center

        if not point0 or not point1:
            return

        self.estimated_distance = (point1 - point0).norm() * chunk.transform.scale
        if scalebar.reference.distance:
            self.reference_distance = scalebar.reference.distance
            self.error_distance = self.estimated_distance - self.reference_distance

    def printScalar(self, f, name, value, precision):
        fmt = "{:." + str(precision) + "f}"
        fmt = "    " + name + ": " + fmt + "\n"
        f.write(fmt.format(value))

    def write(self, f):
        f.write(self.scalebar.label + "\n")
        if self.reference_distance:
            self.printScalar(f, "   distance source", self.reference_distance, 6)
        if self.error_distance:
            self.printScalar(f, "   distance error", self.error_distance, 6)
        if self.estimated_distance:
            self.printScalar(f, "   distance estimated", self.estimated_distance, 6)


def save_estimated_reference():
    filename = Metashape.app.getSaveFileName(filter="*.txt")

    chunk = Metashape.app.document.chunk
    if chunk is None:
        raise Exception("Empty project!")

    with open(filename, "w") as f:
        for camera in chunk.cameras:
            if camera.type != Metashape.Camera.Type.Regular:
                continue

            if not camera.transform:
                continue

            stats = CameraStats(camera)
            stats.write(f)

        for marker in chunk.markers:
            if marker.type != Metashape.Marker.Type.Regular:
                continue

            if not marker.position:
                continue

            stats = MarkerStats(marker)
            stats.write(f)

        for scalebar in chunk.scalebars:
            stats = ScalebarStats(scalebar)
            stats.write(f)


label = "Scripts/Save estimated reference"
Metashape.app.addMenuItem(label, save_estimated_reference)
print("To execute this script press {}".format(label))
