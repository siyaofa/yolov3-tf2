import os
import shutil
path = r'D:\VoTT\raw_images\success'

dst_path=r'D:\VoTT\raw\success'

index=0

for root, dirs, files in os.walk(path):
	for file in files:
		if 'raw' in file:
			index+=1
			file_path=os.path.join(root,file)
			print('root',root)
			parent=os.path.split(root)[-1]
			# print(parent+'_'+file)
			dst_file_path=os.path.join(dst_path,parent+'_'+file)
			shutil.copyfile(file_path,dst_file_path)