import os, zipfile, pickle, json, vtk
from enum import Enum
from dataHandler import FileName
from vtk.util import numpy_support
import numpy as np

class PackDecompress():
    def __init__(self, packets_names, dst_folder, file_num_beg = 0, ):
        self.packets_names_list = packets_names
        self.pack_name_list_size = len(self.packets_names_list)

        self.new_file_num_begin = file_num_beg

        self.dst_folder = dst_folder


    def create_dst_folder(self):
        try:
            if not os.path.exists(self.dst_folder):
                os.makedirs(self.dst_folder)
        except Exception:
            print("creat {} failed".format(self.dst_folder))


    def get_name_ext(self, names):
        path_name_ext = names
        name_ext = os.path.split(path_name_ext)[1]
        return name_ext
    
    def get_ext_name(self, names):
        return os.path.splitext(names)[1]


    def get_zip_pack(self):
        for index, packet_name in enumerate(self.packets_names_list):
            try:
                pack_file = zipfile.ZipFile(packet_name, 'r',allowZip64=True)
            except Exception as e:
                print('pack :%s can not open , igonre \r\n detail:%s' % (packet_name, str(e)))
            else:
                print("handle {}:{}/{}".format(self.get_name_ext(packet_name), index+1, self.pack_name_list_size))
                yield pack_file
        return

    def get_file_number_name(self):
        newfile_name = "{:0>8d}.dcm".format(self.new_file_num_begin)
        self.new_file_num_begin = self.new_file_num_begin + 1
        return newfile_name


    def write_new_record_as_json(self, record):
        """
        transform the python pickle obj to json object
        Then write the json file to dst_folder
        """

        TotalLabel = {}
        for image_name, labels in record.items():
            labelDic = {}

            keys = list(labels.keys())
            # old bugs don't worry
            starts_points = labels[FileName.start_point.start_point]
            ends_points = labels[FileName.end_point]
            HasCrack = labels[FileName.Has_crack.Has_crack]
            
            Points_array = []
            for index, _ in enumerate(starts_points):
                Points_array.append(starts_points[index])
                Points_array.append(ends_points[index])
            PointsNum = len(Points_array)

            labelDic["HasCrack"] = HasCrack
            labelDic["Points"] = Points_array
            labelDic["PointsNum"] = PointsNum
            TotalLabel[image_name] = labelDic

        new_record_path = os.path.join(self.dst_folder, "record.json")
        with open(new_record_path, "w") as f:
            json.dump(TotalLabel, f)
        
        
    def merge_and_make_data_set(self, verbose=False):
        self.create_dst_folder()
        new_global_label_dic = {}
        for pack_index, pack_file in enumerate(self.get_zip_pack()):
            map_path2name_ext = {}
            record_dic = {}
            for x in pack_file.namelist():
                ext_name = self.get_ext_name(x)
                if ext_name == '.dcm':
                    map_path2name_ext[x] = self.get_name_ext(x)
                elif ext_name == '.ibrainRc':
                    record = pack_file.read(x)
                    record_dic = pickle.loads(record)
            
            for old_path, old_name in map_path2name_ext.items():
                if verbose:
                    print("decompress {}".format(old_path))
                try:
                    image_file = pack_file.read(old_path)
                    new_file_name = self.get_file_number_name()
                    new_file_path = os.path.join(self.dst_folder, new_file_name)
                    with open(new_file_path, 'wb+') as file:
                        file.write(image_file)
                    new_global_label_dic[new_file_name] = record_dic[old_name]
                except Exception:
                    print("Corrupt file {}".format(old_name))
                    self.new_file_num_begin = self.new_file_num_begin - 1
            if verbose:
                print("package:{} done create {} images".format(pack_index,
                                                            (str(len(map_path2name_ext.keys()) + 1))
                                                            )
                      )
        
        self.write_new_record_as_json(new_global_label_dic)


def Decompress(src_folder, dst_folder):
    get_ext_name = lambda x: os.path.splitext(x)[1]
    packets_names = [x for x in os.listdir(src_folder) if get_ext_name(x) == ".zip"]
    
    packets_path = [os.path.join(src_folder, x) for x in packets_names]

    pm =PackDecompress(packets_path, dst_folder)
    #pm.merge_and_make_data_set(verbose=True)
    pm.merge_and_make_data_set()


def dcm2npMatrix(filename):
    dcmReader = vtk.vtkDICOMImageReader()
    dcmReader.SetFileName(filename)
    dcmReader.SetDataByteOrderToLittleEndian()
    dcmReader.Update()
    _extent = dcmReader.GetDataExtent()
    dims = [_extent[1] - _extent[0] + 1, _extent[3] - _extent[2] + 1]
    ConstPixelSpacing = dcmReader.GetPixelSpacing()
    imageData = dcmReader.GetOutput()
    pointData = imageData.GetPointData()
    assert (pointData.GetNumberOfArrays() == 1)
    arrayData = pointData.GetArray(0)
    ArrayDicom = numpy_support.vtk_to_numpy(arrayData)
    ArrayDicom = ArrayDicom.reshape(dims, order='F')
    return np.flipud(ArrayDicom.T)

def max_min_normal(im, maxd=255):
    print('using_max_min normal')
    minI = np.min(im)
    im -= minI
    maxI = np.max(im)
    im = im/maxI * maxd
    return im


def histeq(im, nbr_bins=2048):
    '''
    直方图均衡化
    '''
    imhist, bins = np.histogram(im.flatten(), nbr_bins, normed=True)
    cdf = imhist.cumsum()
    cdf = 255 * cdf/(cdf[-1])
    # 使用累积分布函数进行线性插值
    im2 = np.interp(im.flatten(), bins[:-1], cdf)
    return im2.reshape(im.shape)


def make_dcm_data_set():
    SrcRoot = lambda x : os.path.join("./data", x)
    DstRoot = lambda x : os.path.join("./dcms", x)

    test_src_path = SrcRoot("test")
    train_src_path = SrcRoot("train")
    validate_src_path = SrcRoot("validate")
   
    test_dst_path = DstRoot("test")
    train_dst_path = DstRoot("train")
    validate_dst_path = DstRoot("validate")
    
    src_path = [test_src_path, train_src_path, validate_src_path]
    dst_path = [test_dst_path, train_dst_path, validate_dst_path]
    
    for src, dst in zip(src_path, dst_path):
        print("Handling {} -> {}".format(src, dst))
        Decompress(src, dst)


if __name__ == "__main__":
    make_dcm_data_set()