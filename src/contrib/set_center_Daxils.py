import Metashape

def CS_to_camera_Daxils():
	
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

    R = camera.transform #4*4の座標。チャンク座標。カメラの同次座標。
    chunk.transform.translation = Metashape.Vector((0,0,0))
    chunk.transform.matrix = chunk.transform.matrix * R.inv()
    print(chunk.transform.matrix)

    print("System set to " + camera.label + " coordinates. Script finished.")
    return 1
CS_to_camera_Daxils()

Metashape.app.addMenuItem("Custom menu/Coordinate system Daxils", CS_to_camera_Daxils)
#Metashape.app.addMenuItem("Custom menu/Coordinate system to camera", CS_to_camera)
