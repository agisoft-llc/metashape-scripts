"https://www.agisoft.com/forum/index.php?topic=11120.msg50024#msg50024"


import Metashape

""" #Daxils version
def CS_to_camera():
	
	print("Script started...")
	chunk = Metashape.app.document.chunk
	if not chunk:
		print("Empty project, script aborted")
		return
	selected = [camera for camera in chunk.cameras if camera.selected and camera.transform]
	if len(selected) != 1:
		print("Select only one aligned camera to procees. Script aborted.")
		return
	camera = selected[0]

	R = camera.transform

	chunk.transform.translation = Metashape.Vector((0,0,0))
	chunk.transform.matrix = chunk.transform.matrix * R.inv()

	print("System set to " + camera.label + " coordinates. Script finished.")
	return 1
CS_to_camera()
Metashape.app.addMenuItem("Custom menu/Coordinate system to camera", CS_to_camera)
"""

""" #Alex version
def CS_to_camera():
	
	print("Script started...")
	chunk = Metashape.app.document.chunk
	if not chunk:
		print("Empty project, script aborted")
		return
	selected = [camera for camera in chunk.cameras if camera.selected and camera.transform and (camera.type == Metashape.Camera.Type.Regular)]
	if len(selected) != 1:
		print("Select only one aligned camera to procees. Script aborted.")
		return
	camera = selected[0]
	T = chunk.transform.matrix
	origin = (-1) * camera.center
	
	R = Metashape.Matrix().Rotation(camera.transform.rotation()*Metashape.Matrix().Diag([1,-1,1]))
	origin = R.inv().mulp(origin)
	chunk.transform.matrix = T.scale() * Metashape.Matrix().Translation(origin) * R.inv()

	Metashape.app.update()
	

	print("System set to " + camera.label + ". Script finished.")
	return 1
CS_to_camera()

Metashape.app.addMenuItem("Custom menu/Coordinate system to camera", CS_to_camera)	
"""

""" #andyroo version
doc = Metashape.app.document
chunk = doc.chunk
camera = chunk.cameras[2]
#below rotates and translates everything to camera frame of reference
T = chunk.transform.matrix
origin = (-1) * camera.center
R = Metashape.Matrix().Rotation(camera.transform.rotation()*Metashape.Matrix().Diag([1,-1,1]))
origin = R.inv().mulp(origin)
chunk.transform.matrix = Metashape.Matrix().Translation(origin) * R.inv()
"""