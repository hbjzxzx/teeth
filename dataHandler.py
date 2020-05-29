import os, zipfile, pickle, vtk
import numpy as np
from vtk.util import numpy_support
from enum import Enum
from PIL import Image

class FileName(Enum):
    Total_num = 'Total_num.field_name.iBRAIN'
    Has_crack = 'Has_crack.field_name.iBRAIN'
    Has_line = 'Has_line.field_name.iBRAIN'
    start_point = 'start_point.field_name.iBRAIN'
    end_point = 'end_point.field_name.iBRAIN'


class PackManger():

    def __init__(self, root, packages_list, global_record_dir='./boundleData'):
        self.pack_list = [os.path.join(root, x) for x in packages_list]
        self.pack_list_size = len(packages_list)

        self.uid = 0
        self.global_record_dir = global_record_dir
        self.uidprefix = 'ibrainDS'

        if not os.path.exists(self.global_record_dir):
            os.mkdir(self.global_record_dir)

    def merge_and_make_data_set(self, verbose=False):
        new_global_label_dic = {}
        for pack_index, pack_file in enumerate(self.get_zip_pack()):
            image_file_name_list = []
            image_path_list = []
            record_dic = {}
            for x in pack_file.namelist():
                if os.path.splitext(x)[1] == '.dcm':
                    image_file_name_list.append(os.path.split(x)[1])
                    image_path_list.append(x)
                elif os.path.splitext(x)[1] == '.ibrainRc':
                    record = pack_file.read(x)
                    record_dic = pickle.loads(record)
            for image_name, image_path in zip(image_file_name_list, image_path_list):
                image_file = pack_file.read(image_path)
                new_file_name = self.get_uid_path()
                with open(new_file_name, 'wb+') as file:
                    file.write(image_file)
                new_global_label_dic[os.path.split(new_file_name)[1]] = record_dic[image_name]
            if verbose:
                print("package:%s done create %s images" % (pack_index,
                                                            (str(len(image_file_name_list) + 1))
                                                            )
                      )
        with open(self.get_record_file_path(), 'wb') as grecord:
            pickle.dump(new_global_label_dic, grecord)

    def get_zip_pack(self):
        for index, package_path in enumerate(self.pack_list):
            try:
                pack_file = zipfile.ZipFile(package_path, 'r',allowZip64=True)
            except Exception as e:
                print('pack :%s can not open , igonre \r\n detail:%s' % (package_path, str(e)))
                if index != self.pack_list_size - 1:
                    continue
                else:
                    raise StopIteration
            else:
                print(package_path)
                yield pack_file
        raise StopIteration

    def make_data_bundle(self, width, height, verbose=False):
        name_list = []
        record_dit = {}
        for filename in os.listdir(self.global_record_dir):
            if os.path.isfile(os.path.join(self.global_record_dir, filename)):
                if os.path.splitext(filename)[1] == '.dcm':
                    name_list.append(filename)
                elif os.path.splitext(filename)[1] == '.ibrainRc':
                    with open(os.path.join(self.global_record_dir, filename), 'rb') as file:
                        record_dit = pickle.load(file)
        if verbose:
            cnt_total = len(name_list)

        number_of_data = len(name_list)
        channel = 1

        image_data_all = np.zeros((number_of_data, height, width, channel), np.int32)
        flag_label = np.zeros((number_of_data, 1), np.bool)

        self.maxline = 10
        # each line has 2 points and 4 coordinate data plus and each pic label has ending flag -1
        self.maxlinelabelDim = 4 * self.maxline + 1

        # max number of label line in each image is 10 ,
        # data always end with -1 indicate the length of points data
        points_label = np.zeros((number_of_data, 4 * self.maxline + 1), np.int32)

        for index, name in enumerate(name_list):
            image_raw_data = self.read_images(name)
            trans_data, new_start_points, new_end_points = self.transform_to_fix_size(image_raw_data, width, height,
                                                                                      record_dit[name][FileName.start_point.start_point],
                                                                                      record_dit[name][FileName.end_point])
            flag = record_dit[name][FileName.Has_crack.Has_crack]
            line_label_array, dim = self.get_numpy_form_label(new_start_points, new_end_points)

            image_data_all[index, :, :, 0] = trans_data
            flag_label[index, 0] = flag
            points_label[index, 0:dim] = line_label_array
            if verbose:
                if index % 100 == 0:
                    print('progress:%s/%s' % (index, cnt_total))

        # save data using save() method of numpy array
        data_path, flag_path, line_path = self.get_numpy_record_path(width, height)

        error_ = []
        for i in range(flag_label.shape[0]):
            lines = points_label[i]
            for x in lines:
                if x == -1:
                    break
                elif x < 0:
                    error_.append(i)
                    break
        print('washing data...')
        print('before errors {}'.format(len(error_)))
        self.wash(flag_label, points_label)
        error_ = []
        for i in range(flag_label.shape[0]):
            lines = points_label[i]
            for x in lines:
                if x == -1:
                    break
                elif x < 0:
                    error_.append(i)
                    break
        print('after errors {}'.format(len(error_)))
        np.save(data_path, image_data_all)
        np.save(flag_path, flag_label)
        np.save(line_path, points_label)
        if verbose:
            print('done! image size is %s * %s' % (width, height))

    def transform_to_fix_size(self, raw_image, height, width, start_points, end_points):
        raw_H, raw_W = raw_image.shape[0], raw_image.shape[1]
        im = Image.fromarray(raw_image)
        im = im.resize((width, height))

        rate_H, rate_W = height / raw_H, width / raw_W

        new_start_points = []
        new_end_points = []

        for s, e in zip(start_points, end_points):
            new_s = (s[0] * rate_W, s[1] * rate_H)
            new_e = (e[0] * rate_W, e[1] * rate_H)
            new_start_points.append(new_s)
            new_end_points.append(new_e)
        return np.array(im), new_start_points, new_end_points

    def get_numpy_form_label(self, start_points, end_points):
        points = []
        for i in range(len(start_points)):
            s = start_points[i]
            e = end_points[i]
            points.append(s[0]), points.append(s[1])
            points.append(e[0]), points.append(e[1])
        assert len(points) < self.maxlinelabelDim
        points.append(-1)
        return np.array(points, np.int32), len(points)

    def get_uid_path(self):
        now = self.uid
        self.uid += 1
        return os.path.join(self.global_record_dir, "{}{:0>8d}.{}".format(self.uidprefix, now , 'dcm'))

    def get_record_file_path(self):
        return os.path.join(self.global_record_dir, 'label' + '.ibrainRc')

    def get_numpy_record_path(self, width, height):
        root_path = os.path.join(self.global_record_dir, 'numpy_form_record_file_%s_%s' % (width, height))
        if not os.path.exists(root_path):
            os.mkdir(root_path)
        data_path = os.path.join(root_path, 'raw_data.npy')
        flag_path = os.path.join(root_path, 'flag_data.npy')
        line_path = os.path.join(root_path, 'line_data.npy')
        return data_path, flag_path, line_path

    def read_images(self, filename):
        path = os.path.join(self.global_record_dir, filename)
        dcmReader = vtk.vtkDICOMImageReader()
        dcmReader.SetFileName(path)
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



    def wash(self, flag, lines):
        check_stride = 4
        total_num = len(flag)
        for index in range(total_num):
            if flag[index]:
                chk_s = 0
                while lines[index][chk_s] != -1:
                    for line_index in range(chk_s,chk_s+check_stride):
                        if lines[index][line_index] < 0:
                            lines[index][chk_s:chk_s+check_stride] = -2
                    chk_s += check_stride
                final_lines = []
                for i in range(lines[index].shape[0]):
                    if (lines[index][i] != -2) and (lines[index][i] != -1) and (lines[index][i] != -0):
                        final_lines.append(lines[index][i])
                final_lines.append(-1)
                for i in range(41 - len(final_lines)):
                    final_lines.append(0)
                # print('before')
                # print(lines[index])
                # print('after')
                # print(final_lines)
                lines[index] = np.array(final_lines)
                if lines[index][0] == -1:
                    flag[index] = False
            


class input_data_helper():
    '''
    input data function
    '''
    def input_bundle_data(root, one_hot=True, predeal=None):
        raw_data_path = os.path.join(root, 'raw_data.npy')
        flag_data_path = os.path.join(root, 'flag_data.npy')
        line_data_path = os.path.join(root, 'line_data.npy')

        raw_data = np.load(raw_data_path)
        raw_data = raw_data.astype(np.float)

        line_data = np.load(line_data_path)
        line_data = line_data.astype(np.float)

        flag_data = np.load(flag_data_path)
        if not one_hot:
            flag_data = flag_data.astype(np.float)
            if predeal:
                for s in range(len(raw_data)):
                    raw_data[s] = predeal(raw_data[s])
            return raw_data, flag_data, line_data

        temp_flag_data = np.zeros((len(flag_data), 2), dtype=np.float)
        for i in range(flag_data.shape[0]):
            if flag_data[i]:
                temp_flag_data[i][0] = 1
            else:
                temp_flag_data[i][1] = 1
        if predeal:
            for s in range(len(raw_data)):
                raw_data[s] = predeal(raw_data[s])
        return raw_data, temp_flag_data, line_data

    def get_train_validate_test(images, flag_labels, line_labels, valida_num, test_num):
        total_num = images.shape[0]
        

        positive_index_array = np.nonzero(flag_labels[:,0])[0]
        negative_index_array = np.nonzero(flag_labels[:,1])[0]
        
        positive_index_array_array = np.random.choice(positive_index_array.shape[0], int((valida_num + test_num)/2), replace=False)
        negative_index_array_array = np.random.choice(negative_index_array.shape[0], int((valida_num + test_num)/2), replace = False)

        positive_index_array = positive_index_array[positive_index_array_array]
        negative_index_array = negative_index_array[negative_index_array_array]

        train_data_index = np.ones((total_num), dtype=np.bool)
        train_data_index[positive_index_array] = False
        train_data_index[negative_index_array] = False

        anchor = int(valida_num/2)
        validate_data_index = np.zeros((total_num), dtype=np.bool)
        validate_data_index[positive_index_array[:anchor]] = True
        validate_data_index[negative_index_array[:anchor]] = True

        test_data_index = np.zeros((total_num), dtype=np.bool)
        test_data_index[positive_index_array[anchor:]] = True
        test_data_index[negative_index_array[anchor:]] = True

        assert np.sum(train_data_index) + np.sum(validate_data_index) + np.sum(test_data_index) == total_num

        train_data, train_flag, train_lines = images[train_data_index], flag_labels[train_data_index], line_labels[train_data_index]
        
        validate_date, validate_flag, validate_lines = images[validate_data_index], flag_labels[validate_data_index], line_labels[validate_data_index]

        test_data, test_flag, test_lines = images[test_data_index], flag_labels[test_data_index], line_labels[test_data_index]
        return (train_data, train_flag, train_lines), (validate_date, validate_flag, validate_lines), (test_data, test_flag, test_lines)
    def bootstrap(raw_data, flag_data, line_data):
        np.random.seed(20)
        _size = len(raw_data)

        train_index = np.zeros((_size,), np.bool)
        for i in range(_size):
            rand_index = np.random.randint(0, _size, size=1)[0]
            train_index[rand_index] = True
        raw_data_train = raw_data[train_index]
        flag_data_train = flag_data[train_index]
        line_data_train = line_data[train_index]

        test_index = ~train_index

        raw_data_test = raw_data[test_index]
        flag_data_test = flag_data[test_index]
        line_data_test = line_data[test_index]

        return (raw_data_train, flag_data_train, line_data_train,
                raw_data_test, flag_data_test, line_data_test)

    def max_min_normal(im):
        print('using_max_min normal')
        maxI = np.max(im)
        minI = np.min(im)
        rangeI = maxI - minI
        im -= minI
        im = im/maxI *255
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


    def fft(im):
        temp = im
        im = im.reshape(200,200)
        f = np.fft.fft2(im)
        fshift = np.fft.fftshift(f)
        rows, cols = im.shape
        mask = np.ones(im.shape, np.uint8)

        margin = 1
        mask[rows // 2 - margin: rows // 2 + margin, cols // 2 - margin: cols // 2 + margin] = 0

        high = 50
        mask[0:high] = 0
        mask[-high:] = 0
        mask[:, 0:high] = 0
        mask[:, -high:] = 0

        fshift2 = fshift * mask
        f2 = np.fft.ifftshift(fshift2)
        new_image = np.fft.ifft2(f2)
        new_image = np.abs(new_image)
        return new_image.reshape(temp.shape)


    def read_from_png(root, label=True):
        images_name = os.listdir(root)
        number = len(images_name)

        first_im = Image.open(os.path.join(root,images_name[0]))
        width, height = first_im.size
        images = np.zeros((number, width, height,1), np.int32)
        labels = np.zeros((number, 2), np.bool)

        if label:
            labels[:, 0] = True
        else:
            labels[:, 1] = True
            
        for index, im_name in enumerate(images_name):
            path = os.path.join(root,im_name)
            im = Image.open(path)

            images[index, :, :, 0] = np.array(im)
        return images, labels, number
