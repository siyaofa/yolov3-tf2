import os
import zipfile


'''
按照阿里云服务器上的目录解析指定的内容

'''


def unzip(src, dst):
    print(src, dst)
    f = zipfile.ZipFile(src, 'r')
    for file in f.namelist():
        f.extract(file, dst)


unzip_path_root = r'D:\VoTT\raw_images\success'
src_path_root = r'D:\BaiduNetdiskDownload\魔方助手APP收集数据\raw_images\success\raw'
unzip_cube_order = 3

zip_filenames = []


dates = os.listdir(src_path_root)
for date in dates:
    date_path = os.path.join(src_path_root, date)
    devices = os.listdir(date_path)
    for device in devices:
        device_path = os.path.join(date_path, device)
        cube_orders = os.listdir(device_path)
        for cube_order in cube_orders:
            if(cube_order == str(unzip_cube_order)):
                cube_order_path = os.path.join(device_path, cube_order)
                filenames = os.listdir(cube_order_path)
                for filename in filenames:
                    zip_path = os.path.join(cube_order_path, filename)
                    try:
                        unzip(zip_path, unzip_path_root+'\\' +
                              os.path.splitext(filename)[0]+'_'+device+'_'+str(unzip_cube_order))
                    except:
                        print(zip_path + ' unzip failed ')
                        pass
