import Metashape

chunk = Metashape.app.document.chunk
cameras = chunk.cameras

c = Metashape.Vector([0,0,0])
cs = Metashape.Vector([0,0,0])
n = 0
T = chunk.transform.matrix  #変換状態を調べる
crs = chunk.crs #座標系を調べる

for camera in cameras:
    c = camera.center #カメラセンターの座標で
    cs = cs + c
    n = n + 1
    #print(n)
CCMass = cs/n
#print(cs)
#print(n)
#print(CCMass)

coord = crs.project(T.mulp(CCMass)) #Center of whole cameras

R = camera.transform #4*4の座標。カメラの同次座標。チャンク座標
chunk.transform.translation = Metashape.Vector((0,0,0))
chunk.transform.matrix = chunk.transform.matrix * R.inv()


"""Alex ver
origin = (-1)*CCMass

R = Metashape.Matrix().Rotation(camera.transform.rotation()*Metashape.Matrix().Diag([1,1,1]))
origin = R.inv().mulp(origin)

chunk.transform.matrix = T.scale() * Metashape.Matrix().Translation(origin) * R.inv()

Metashape.app.update()
"""

""" for chunks
for chunk in chunks:
    cameras = chunk.cameras
    for camera in cameras:
        c = [0,0,0]
        cs = [0,0,0]
        n = 0
        c = camera.center
        cs = cs + c
        n = n + 1
    CCMass = cs/n
    print(CCMass)
"""