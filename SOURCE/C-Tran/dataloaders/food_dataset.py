# FoodData.py
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

import json, string
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# import nltk

# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')

from dataloaders.data_utils import get_unk_mask_indices
from dataloaders.data_utils import image_loader, image_loader_food
from torchvision import transforms


def crop_image_by_mask(mask, width, height):
    array_2d = mask.transpose(1,2,0).reshape(-1, mask.shape[0])
    # After transpose x -> y, y -> x. So Row = x, y = col

    row_array, col_array = np.where(array_2d == 1)
    x_array = row_array
    y_array = col_array
    min_x, min_y = x_array.min(), y_array.min()
    max_x, max_y = x_array.max(), y_array.max()

    width_scale = width/512
    height_scale = height/512

    min_x = float(min_x*width_scale)
    min_y = float(min_y*height_scale)
    max_x = float(max_x*width_scale)
    max_y = float(max_y*height_scale)

    return (min_x, min_y, max_x, max_y)


def buildVoc(objData):
    spunctuation = set(string.punctuation)
    swords = set(stopwords.words('english'))
    print('Building vocabulary of words...')
    lem = WordNetLemmatizer()
    word_counts = dict()

    # Fix outlier
    objData['categories'][0]['name_readable'] = re.sub(r'[,\s]+', '_', objData['categories'][0]['name_readable'])
    
    word_counts = dict()
    for (i, entry) in enumerate(objData['categories']):
        for word in word_tokenize(entry['name_readable'].lower()): # Get token of word
            word = lem.lemmatize(word) # -> Take the word to bare word
            if word not in swords and word not in spunctuation:
                word_counts[word] = 1 + word_counts.get(word, 0)

    sword_counts = sorted(word_counts.items(), key = lambda x: -x[1])
    id2word = {idx: word for (idx, (word, count)) in enumerate(sword_counts)}
    id2count = {idx: count for (idx, (word, count)) in enumerate(sword_counts)}
    word2id = {word: idx for (idx, word) in id2word.items()}
    vocabulary = (id2word, word2id, id2count)

    return vocabulary


def buidlLabel(objData, vocabulary):
    imageIds = [entry['id'] for entry in objData['images']]
    imageId2index = {image_id: idx for (idx, image_id) in enumerate(imageIds)}

    lem = WordNetLemmatizer()
    labels = np.zeros((len(objData['images']), len(vocabulary[0])))
    image_names = []
    for entry in objData['images']:
        caption = 'null'
        for val in objData['annotations']:
            if val['image_id'] == entry['id']:
                key = val['category_id']
                for check in objData['categories']:
                    if check['id'] == key:
                        caption = check['name_readable']
                        break
                image_id = entry['id']
                for word in word_tokenize(caption.lower()):
                    word = lem.lemmatize(word)
                    if word in vocabulary[1].keys():
                        labels[imageId2index[image_id], vocabulary[1][word]] = 1

        image_names.append(entry['file_name'])

    image_names = np.array(image_names)

    return labels, image_names


class FoodDataset(Dataset):
    def __init__(self, trainRoot, img_dir, label_dir, label_obs_dir, image_names_dir, image_transform, known_labels = 0, testing = False, seg_model = None, seg = False): ###
        self.img_dir = img_dir
        self.image_transform = image_transform

        if os.path.exists(label_dir) and os.path.exists(image_names_dir):
            print('Loading labels and image names')
            self.labels = np.load(label_dir)
            self.labels_obs = np.load(label_obs_dir) ###
            self.names = np.load(image_names_dir)
        else:
            # Build voca
            print("Build labels ...")
            annotation_dir = os.path.join(trainRoot, "annotations.json")
            objData = json.load(open(annotation_dir))
            vocabulary = buildVoc(objData)

            # Save path
            savePath = os.path.join(trainRoot, 'labels.npy')
            savePathName = os.path.join(trainRoot, 'image_names.npy')

            # Build label
            self.labels, self.names = buidlLabel(objData, vocabulary)

            # Save Label and Name of image
            np.save(savePath, self.labels)
            np.save(savePathName, self.names)

        self.num_labels = len(self.labels[0])
        self.known_labels = known_labels
        self.testing = testing
        self.seg = seg
        self.seg_model = seg_model

    def __len__(self):
        return len(self.names)


    def __getitem__(self, index):

        name = self.names[index]

        img_path = os.path.join(self.img_dir, name)

        # Add module segmentation
        image, width, height = image_loader_food(img_path)

        if self.seg:
            device = "cuda" if torch.cuda.is_available() else "cpu"

            transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor()])

            img = transform(image).float().to(device)
            img = img.unsqueeze(0)

            pred_mask = self.seg_model(img)

            pred_mask = pred_mask.squeeze(0).cpu().detach()
            pred_mask = pred_mask.permute(1, 2, 0)
            pred_mask[pred_mask < 0] = 0
            pred_mask[pred_mask > 0] = 1

            mask = pred_mask.numpy()
            crop_box = crop_image_by_mask(mask, width, height)
            image = image.crop(crop_box)
        
        if self.image_transform is not None:
            image = self.image_transform(image)


        label = torch.Tensor(self.labels[index])
        label_obs = torch.Tensor(self.labels_obs[index]) ###

        unk_mask_indices = get_unk_mask_indices(image, self.testing, self.num_labels, self.known_labels)
        mask = label.clone()
        mask.scatter_(0, torch.Tensor(unk_mask_indices).long(), -1)

        sample = {}
        sample['image'] = image
        sample['labels'] = label
        sample['labels_obs'] = label_obs ###
        sample['mask'] = mask
        sample['imageIDs'] = name
        sample['idx'] = index ###

        return sample
