import torch
# from skimage import io, transformclear
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pdb import set_trace as stop
import os, random
from dataloaders.voc2007_20 import Voc07Dataset
from dataloaders.vg500_dataset import VGDataset
from dataloaders.coco80_dataset import Coco80Dataset
from dataloaders.news500_dataset import NewsDataset
from dataloaders.coco1000_dataset import Coco1000Dataset
from dataloaders.cub312_dataset import CUBDataset
from dataloaders.food_dataset import FoodDataset
from dataloaders.unet import UNet
import warnings
warnings.filterwarnings("ignore")


def get_data(args):
    dataset = args.dataset
    data_root=args.dataroot
    batch_size=args.batch_size

    rescale=args.scale_size
    random_crop=args.crop_size
    attr_group_dict=args.attr_group_dict
    workers=args.workers
    n_groups=args.n_groups
    
    seg_model = None
    if args.use_seg:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_pth = '/media/btlen02/paper/C-Tran/segmentation_models/unet.pth'
        model = UNet(in_channels = 3, num_classes = 1).to(device)
        model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))
        seg_model = model

    normTransform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if args.test_batch_size == -1:
        args.test_batch_size = batch_size

    # Transform with cropped images
    scale_size = rescale
    crop_size = random_crop
    trainTransform = transforms.Compose([transforms.Resize((crop_size, crop_size)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        normTransform])

    testTransform = transforms.Compose([transforms.Resize((crop_size, crop_size)),
                                        transforms.ToTensor(),
                                        normTransform])


    # Original transform
    scale_size = rescale
    crop_size = random_crop
    trainTransform = transforms.Compose([transforms.Resize((scale_size, scale_size)),
                                        transforms.RandomChoice([
                                        transforms.RandomCrop(640),
                                        transforms.RandomCrop(576),
                                        transforms.RandomCrop(512),
                                        transforms.RandomCrop(384),
                                        transforms.RandomCrop(320)
                                        ]),
                                        transforms.Resize((crop_size, crop_size)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        normTransform])

    testTransform = transforms.Compose([transforms.Resize((scale_size, scale_size)),
                                        transforms.CenterCrop(crop_size),
                                        transforms.ToTensor(),
                                        normTransform])



    # New transform
    # scale_size = 700
    # crop_size = 656

    # trainTransform = transforms.Compose([transforms.Resize((scale_size, scale_size)),
    #                                     transforms.RandomChoice([
    #                                     transforms.RandomCrop(700), #640
    #                                     transforms.RandomCrop(656), #576
    #                                     transforms.RandomCrop(600), #512
    #                                     transforms.RandomCrop(576), #384
    #                                     transforms.RandomCrop(524)  #320
    #                                     ]),
    #                                     transforms.Resize((crop_size, crop_size)),
    #                                     transforms.RandomHorizontalFlip(),
    #                                     transforms.ToTensor(),
    #                                     normTransform
    #                                     ])

    # testTransform = transforms.Compose([transforms.Resize((scale_size, scale_size)),
    #                                     transforms.CenterCrop(crop_size),
    #                                     transforms.ToTensor(),
    #                                     normTransform])

    test_dataset = None
    test_loader = None
    drop_last = False
    if dataset == 'coco':
        coco_root = os.path.join(data_root,'coco')
        ann_dir = os.path.join(coco_root,'annotations_pytorch')
        train_img_root = os.path.join(coco_root,'train2014')
        test_img_root = os.path.join(coco_root,'val2014')
        train_data_name = 'train.data'
        val_data_name = 'val_test.data'
        
        train_dataset = Coco80Dataset(
            split='train',
            num_labels=args.num_labels,
            data_file=os.path.join(coco_root,train_data_name),
            img_root=train_img_root,
            annotation_dir=ann_dir,
            max_samples=args.max_samples,
            transform=trainTransform,
            known_labels=args.train_known_labels,
            testing=False)
        valid_dataset = Coco80Dataset(split='val',
            num_labels=args.num_labels,
            data_file=os.path.join(coco_root,val_data_name),
            img_root=test_img_root,
            annotation_dir=ann_dir,
            max_samples=args.max_samples,
            transform=testTransform,
            known_labels=args.test_known_labels,
            testing=True)

    elif dataset == 'coco1000':
        ann_dir = os.path.join(data_root,'coco','annotations_pytorch')
        data_dir = os.path.join(data_root,'coco')
        train_img_root = os.path.join(data_dir,'train2014')
        test_img_root = os.path.join(data_dir,'val2014')
        
        train_dataset = Coco1000Dataset(ann_dir, data_dir, split = 'train', transform = trainTransform,known_labels=args.train_known_labels,testing=False)
        valid_dataset = Coco1000Dataset(ann_dir, data_dir, split = 'val', transform = testTransform,known_labels=args.test_known_labels,testing=True)
    
    elif dataset == 'vg':
        vg_root = os.path.join(data_root,'VG')
        train_dir=os.path.join(vg_root,'VG_100K')
        train_list=os.path.join(vg_root,'train_list_500.txt')
        test_dir=os.path.join(vg_root,'VG_100K')
        test_list=os.path.join(vg_root,'test_list_500.txt')
        train_label=os.path.join(vg_root,'vg_category_500_labels_index.json')
        test_label=os.path.join(vg_root,'vg_category_500_labels_index.json')

        train_dataset = VGDataset(
            train_dir,
            train_list,
            trainTransform, 
            train_label,
            known_labels=0,
            testing=False)
        valid_dataset = VGDataset(
            test_dir,
            test_list,
            testTransform,
            test_label,
            known_labels=args.test_known_labels,
            testing=True)
    
    elif dataset == 'food':
        food_root = os.path.join(data_root,'food')
        # Train
        train_dir = os.path.join(food_root,'train','images_cropped')
        train_root = os.path.join(food_root,'train')
        train_label = os.path.join(train_root,'labels.npy')
        train_label_obs = os.path.join(train_root,'labels_obs.npy') ###
        train_image_names_dir = os.path.join(train_root,'image_names.npy')
        # Val
        val_dir = os.path.join(food_root,'val','images_cropped')
        val_root = os.path.join(food_root,'val')
        val_label = os.path.join(val_root,'labels.npy')
        val_label_obs = os.path.join(val_root,'labels_obs.npy') ###
        val_image_names_dir = os.path.join(val_root,'image_names.npy')


        # known_labels = args.train_known_labels (100 if use lmt)
        # 25% known = 81 labels
        # 50% known = 162 labels
        # 75% known = 243 labels
        # 0% known = 0 labels
        train_dataset = FoodDataset(
                train_root,
                train_dir,
                train_label,
                train_label_obs,
                train_image_names_dir,
                trainTransform,
                known_labels = 0,
                testing = False,
                seg_model = seg_model,
                seg = args.use_seg)
        
        valid_dataset = FoodDataset(
                val_root,
                val_dir,
                val_label,
                val_label_obs,
                val_image_names_dir,
                testTransform,
                known_labels = 0,
                testing = True,
                seg_model = seg_model,
                seg = args.use_seg)

    elif dataset == 'food_80_20':
        # Modified after 26/3/2024.
        food_root = os.path.join(data_root,'food')
        food_80_20_root = os.path.join(data_root,'food_80_20')
        
        # Train
        train_dir = os.path.join(food_root,'train','images')
        train_root = os.path.join(food_80_20_root,'train')
        train_label_obs = os.path.join(train_root,'labels_obs.npy')
        train_label = os.path.join(train_root,'labels.npy')
        train_image_names_dir = os.path.join(train_root,'image_names.npy')

        # Val
        val_dir = os.path.join(food_root,'train','images')
        val_root = os.path.join(food_80_20_root,'val')
        val_label_obs = os.path.join(val_root,'labels_obs.npy') ###
        val_label = os.path.join(val_root,'labels.npy')
        val_image_names_dir = os.path.join(val_root,'image_names.npy')

        # Test
        test_dir = os.path.join(food_root,'val','images')
        test_root = os.path.join(food_80_20_root,'test')
        test_label_obs = os.path.join(test_root,'labels_obs.npy') ###
        test_label = os.path.join(test_root,'labels.npy')
        test_image_names_dir = os.path.join(test_root,'image_names.npy')


        train_dataset = FoodDataset(
                train_root,
                train_dir,
                train_label,
                train_label_obs,
                train_image_names_dir,
                trainTransform,
                known_labels = 0,
                testing = False,
                seg_model = seg_model,
                seg = args.use_seg)
        
        valid_dataset = FoodDataset(
                val_root,
                val_dir,
                val_label,
                val_label_obs,
                val_image_names_dir,
                testTransform,
                known_labels = 0,
                testing = True,
                seg_model = seg_model,
                seg = args.use_seg)
        
        test_dataset = FoodDataset(
                test_root,
                test_dir,
                test_label,
                test_label_obs,
                test_image_names_dir,
                testTransform,
                known_labels = 0,
                testing = True,
                seg_model = seg_model,
                seg = args.use_seg)

    elif dataset == 'news':
        drop_last=True
        ann_dir = '/bigtemp/jjl5sw/PartialMLC/data/bbc_data/'

        train_dataset = NewsDataset(ann_dir, split = 'train', transform = trainTransform,known_labels=0,testing=False)
        valid_dataset = NewsDataset(ann_dir, split = 'test', transform = testTransform,known_labels=args.test_known_labels,testing=True)
    
    elif dataset=='voc':
        voc_root = os.path.join(data_root,'voc/VOCdevkit/VOC2007/')
        img_dir = os.path.join(voc_root,'JPEGImages')
        anno_dir = os.path.join(voc_root,'Annotations')
        train_anno_path = os.path.join(voc_root,'ImageSets/Main/trainval.txt')
        test_anno_path = os.path.join(voc_root,'ImageSets/Main/test.txt')

        train_dataset = Voc07Dataset(
            img_dir=img_dir,
            anno_path=train_anno_path,
            image_transform=trainTransform,
            labels_path=anno_dir,
            known_labels=args.train_known_labels,
            testing=False,
            use_difficult=False)
        valid_dataset = Voc07Dataset(
            img_dir=img_dir,
            anno_path=test_anno_path,
            image_transform=testTransform,
            labels_path=anno_dir,
            known_labels=args.test_known_labels,
            testing=True)

    elif dataset == 'cub':
        drop_last=True
        resol=299
        resized_resol = int(resol * 256/224)
        
        trainTransform = transforms.Compose([
            #transforms.Resize((resized_resol, resized_resol)),
            #transforms.RandomSizedCrop(resol),
            transforms.ColorJitter(brightness=32/255, saturation=(0.5, 1.5)),
            transforms.RandomResizedCrop(resol),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), #implicitly divides by 255
            transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
            ])

        testTransform = transforms.Compose([
            #transforms.Resize((resized_resol, resized_resol)),
            transforms.CenterCrop(resol),
            transforms.ToTensor(), #implicitly divides by 255
            transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
            ])
        
        cub_root = os.path.join(data_root,'CUB_200_2011')
        image_dir = os.path.join(cub_root,'images')
        train_list = os.path.join(cub_root,'class_attr_data_10','train_valid.pkl')
        valid_list = os.path.join(cub_root,'class_attr_data_10','train_valid.pkl')
        test_list = os.path.join(cub_root,'class_attr_data_10','test.pkl')

        train_dataset = CUBDataset(image_dir, train_list, trainTransform,known_labels=args.train_known_labels,attr_group_dict=attr_group_dict,testing=False,n_groups=n_groups)
        valid_dataset = CUBDataset(image_dir, valid_list, testTransform,known_labels=args.test_known_labels,attr_group_dict=attr_group_dict,testing=True,n_groups=n_groups)
        test_dataset = CUBDataset(image_dir, test_list, testTransform,known_labels=args.test_known_labels,attr_group_dict=attr_group_dict,testing=True,n_groups=n_groups)
        
    else:
        print('no dataset avail')
        exit(0)

    # Modified after 26/3/2024.
    train_obs = None
    val_obs = None
    test_obs = None

    if train_dataset is not None:
        train_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True, num_workers=workers,drop_last=drop_last)
        train_obs = np.load(train_label_obs)
    if valid_dataset is not None:
        valid_loader = DataLoader(valid_dataset, batch_size=args.test_batch_size,shuffle=False, num_workers=workers)
        val_obs = np.load(val_label_obs)
    if test_dataset is not None:
        test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size,shuffle=False, num_workers=workers)
        test_obs = np.load(test_label_obs)
    
    # Original
    # return train_loader, valid_loader, test_loader

    # Updated
    return train_loader, valid_loader, test_loader, train_obs, val_obs, test_obs ###
