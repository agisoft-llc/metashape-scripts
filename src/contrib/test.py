import Metashape
import statistics

chunk = Metashape.app.document.chunk
new_region = Metashape.Region()
xcoord = Metashape.Vector([10E10, -10E10])
ycoord = Metashape.Vector([10E10, -10E10])
avg = [[],[]]
T = chunk.transform.matrix  #変換状態を調べる
s = chunk.transform.matrix.scale() #スケールを調べる
crs = chunk.crs #座標系を調べる
z = Metashape.Vector([0,0]) #０ベクトルを作る

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


By
for camera in chunk.cameras:
#get min and max x,y coords of all camera positions
   if camera.transform:
      coord = crs.project(T.mulp(camera.center))
      xcoord[0] = min(coord.x, xcoord[0])
      xcoord[1] = max(coord.x, xcoord[1])
      ycoord[0] = min(coord.y, ycoord[0])
      ycoord[1] = max(coord.y, ycoord[1])
#get median z-coord of sparse points as lowest z value for boundary box
for point in chunk.point_cloud.points:
   medZ[0].append = point.coord.z
z = statistics.median(medZ[0])