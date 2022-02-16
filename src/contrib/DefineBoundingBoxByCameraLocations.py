import Metashape, statistics
BUFFER = 10 #percent

def cross(a, b):
	result = Metashape.Vector([a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y *b.x])
	return result.normalized()

chunk = Metashape.app.document.chunk
new_region = Metashape.Region()
xcoord = Metashape.Vector([10E10, -10E10])
ycoord = Metashape.Vector([10E10, -10E10])
avg = [[],[]]
T = chunk.transform.matrix
s = chunk.transform.matrix.scale()
crs = chunk.crs
z = Metashape.Vector([0,0])

for camera in chunk.cameras:
	if camera.transform:
		coord = crs.project(T.mulp(camera.center))
		xcoord[0] = min(coord.x, xcoord[0])
		xcoord[1] = max(coord.x, xcoord[1])
		ycoord[0] = min(coord.y, ycoord[0])
		ycoord[1] = max(coord.y, ycoord[1])
		z[0] += coord.z
		z[1] += 1
		avg[0].append(coord.x)
		avg[1].append(coord.y)
z = z[0] / z[1]
avg = Metashape.Vector([statistics.median(avg[0]), statistics.median(avg[1]), z])

corners = [Metashape.Vector([xcoord[0], ycoord[0], z]),
			Metashape.Vector([xcoord[0], ycoord[1], z]),
			Metashape.Vector([xcoord[1], ycoord[1], z]),
			Metashape.Vector([xcoord[1], ycoord[0], z])]
corners = [T.inv().mulp(crs.unproject(x)) for x in list(corners)]			

side1 = corners[0] - corners[1]
side2 = corners[0] - corners[-1]
side1g = T.mulp(corners[0]) - T.mulp(corners[1])
side2g = T.mulp(corners[0]) - T.mulp(corners[-1])
side3g = T.mulp(corners[0]) - T.mulp(Metashape.Vector([corners[0].x, corners[0].y, 0]))
new_size = ((100 + BUFFER) / 100) * Metashape.Vector([side2g.norm()/s, side1g.norm()/s, 3*side3g.norm() / s]) ##

xcoord, ycoord, z = T.inv().mulp(crs.unproject(Metashape.Vector([sum(xcoord)/2., sum(ycoord)/2., z - 2 * side3g.z]))) #
new_center = Metashape.Vector([xcoord, ycoord, z]) #by 4 corners

horizontal = side2
vertical = side1
normal = cross(vertical, horizontal)
horizontal = -cross(vertical, normal)
vertical = vertical.normalized()

R = Metashape.Matrix ([horizontal, vertical, -normal])
new_region.rot = R.t()

new_region.center = new_center
new_region.size = new_size
chunk.region = new_region