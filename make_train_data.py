from dataHandler import *
import os

def merge_data_make_bundle(pack_path, pack_list_to_merge, dest_path, width, height):
    pm = PackManger(pack_path, pack_list_to_merge, global_record_dir=dest_path)
    pm.merge_and_make_data_set(verbose=True)
    pm.make_data_bundle(width, height, verbose=True)

def make_data_set(root, dest, width=224, height=224):
    pack_root = root
    pack_name_list = [pack_name for pack_name in os.listdir(pack_root)] 
    dest_path = dest
    if not os.path.isdir(dest_path):
        os.makedirs(dest_path)
        merge_data_make_bundle(pack_root, pack_name_list, dest_path, width, height)
    elif os.listdir(dest_path):
        print('already exit file in dest folder!')
    else :
        merge_data_make_bundle(pack_root, pack_name_list, dest_path, width, height)


if __name__=='__main__':
    train_souce_root = './data/packages2/train'
    validate_souce_root = './data/packages2/validate'
    test_souce_root = './data/packages2/test'
    
    # train_dest_root = './data/vgg_19_data/train'
    # validate_dest_root = './data/vgg_19_data/validate'
    # test_dest_root = './data/vgg_19_data/test'

    # make_data_set(train_souce_root, train_dest_root)
    # make_data_set(validate_souce_root, validate_dest_root)
    # make_data_set(test_souce_root, test_dest_root)


    train_dest_root = './data/vgg_19_data_X2/train'
    validate_dest_root = './data/vgg_19_data_X2/validate'
    test_dest_root = './data/vgg_19_data_X2/test'

    make_data_set(train_souce_root, train_dest_root, width=448, height=448)
    make_data_set(validate_souce_root, validate_dest_root, width=448, height=448)
    make_data_set(test_souce_root, test_dest_root, width=448, height=448)
