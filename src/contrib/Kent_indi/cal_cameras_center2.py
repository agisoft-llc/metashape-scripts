import Metashape

chunk = Metashape.app.document.chunk
cameras = chunk.cameras

c = Metashape.Matrix.Diag((0,0,0,0))
cs = Metashape.Matrix.Diag((0,0,0,0))
n = 0
T = chunk.transform.matrix  #変換状態を調べる
crs = chunk.crs #座標系を調べる

for camera in cameras:
    c = camera.transform #カメラの同次座標で。
    cs = cs + c
    n = n + 1
CCtrans = 1/n*cs
print(cs)
print(n)
print(CCtrans)


#coord = crs.project(T.mulp(CCmass)) #Center of whole cameras　これはうまくいっているみたい
"""
R = camera.transform #4*4の座標。チャンク座標
chunk.transform.translation = Metashape.Vector((0,0,0))
chunk.transform.matrix = chunk.transform.matrix * CCtrans.inv()
"""