import Metashape
import csv
import pprint

compatible_major_version = "1.8"
found_major_version = ".".join(Metashape.app.version.split('.')[:2])
if found_major_version != compatible_major_version:
    raise Exception("Incompatible Metashape version: {} != {}".format(found_major_version, compatible_major_version))

doc = Metashape.app.document
chunks = doc.chunks

with open('I:\Scales.csv') as file:  
	reader = csv.reader(file)
	list = [row for row in reader] #listのなかに行ごとの集合として格納される。
	print(list)

for chunk in chunks:
	if chunk.enabled is True:
		print(str(chunk))
		i = 0
		while i < 24: #24つのスケールバーを使う。一行ずつスケールバー作成を繰り返す。
		
			l = list[i]
			i += 1
			scale1 = None
			scale2 = None
			p1, p2, dist, acc = l[0],l[1],l[2],l[3]
			
			if (len(chunk.markers) > 0):
				
				for marker in chunk.markers: 
					if (marker.label == p1):
						scale1 = marker
					if (marker.label == p2):
						scale2 = marker
				print(scale1)
				print(scale2)
				if scale1 != None and scale2 != None: 
					scalebar = chunk.addScalebar(scale1,scale2)
					scalebar.reference.distance = float(dist) 
					scalebar.reference.accuracy = float(acc)
					
				else:
					continue
		chunk.updateTransform()

#		camera = chunk.cameras[-1]
#		s = chunk.transform.scale
#		#print("scale is "+str(s))
#		origin = (-1) * camera.center
#		#print("the origin is "+str(origin))
#		R = Metashape.Matrix().Rotation(camera.transform.rotation()* Metashape.Matrix().Diag([1, -1, 1]))
#		origin = R.inv().mulp(origin)
#		chunk.transform.matrix = Metashape.Matrix([[s, 0, 0, 0], [0, 0, s, 0], [0, s, 0, 0], [0, 0, 0, 1]]) * Metashape.Matrix().Translation(origin) * R.inv()
					
#file.close()
#Metashape.app.update()