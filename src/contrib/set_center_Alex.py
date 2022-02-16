import Metashape

def CS_to_camera_Alex():
	
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

	T = chunk.transform.matrix		#初期値を保存.一緒。
	origin = (-1) * camera.center
	print(T)

	R = Metashape.Matrix().Rotation(camera.transform.rotation()*Metashape.Matrix().Diag([1,1,1]))
	origin = R.inv().mulp(origin)
	print(R)
	chunk.transform.matrix = T.scale() * Metashape.Matrix().Translation(origin) * R.inv()

	Metashape.app.update()
	

	print("System set to " + camera.label + ". Script finished.")
	return 1
CS_to_camera_Alex()

Metashape.app.addMenuItem("Custom menu/Coordinate system to camera Alex", CS_to_camera_Alex)
