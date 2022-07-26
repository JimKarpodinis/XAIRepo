from random import randint
from tqdm import tqdm
from predictor import read_image_from_url
import requests
'''
This python script contains the custom coco dataset class, used for holding images and their annotations!
'''

def is_sublist(list1, list2):
    l1 = list1.copy()
    l2 = list2.copy()
    for l in l1:
        if l in l2:
            l2.remove(l)
        else:
            return False
    return True

class COCO:
    '''
    Receives a json file, and produces a dictionary containing all relevant information
    about each image.
    '''
    def __init__(self, json_data):
        self.json_data = json_data
        self.categories = {}
        self.coco = {} #the main dictionary
        for im in self.json_data["images"]:
            self.coco[im["id"]] = {"url": im["flickr_url"]} #saving url for each img

        for cat in self.json_data["categories"]: #makes categories dictionary, that contain all possible objects.
            self.categories[cat["id"]] = {"name": cat["name"], "supercategory": cat["supercategory"]}

        anns_per_image = {}
        print ("Loading Coco Metadata")
        for ann in tqdm(self.json_data["annotations"]): #for each image, it creates a list with annotated objects
            if ann["image_id"] not in anns_per_image:
                anns_per_image[ann["image_id"]] = {"label_ids": [ann["category_id"]], "label_texts": [self.categories[ann["category_id"]]["name"]]}
            else:
                anns_per_image[ann["image_id"]]["label_ids"].append(ann["category_id"])
                anns_per_image[ann["image_id"]]["label_texts"].append(self.categories[ann["category_id"]]["name"])

        # we join categories with the main dict of imgs
        for im_id in list(self.coco):
            if im_id in anns_per_image:
                self.coco[im_id]["label_ids"] = anns_per_image[im_id]["label_ids"]
                self.coco[im_id]["label_texts"] = anns_per_image[im_id]["label_texts"]
            else:
                self.coco.pop(im_id, None) # αν δεν έχουμε annotations σβήνουμε το key αυτό

    def prune_dataset(self, labels):
        #receives a label list of objects and drops from dataset
        #all images that don't contain at least once each item in that label list.
        image_ids = list(self.coco.keys())
        for image_id in image_ids:
            if not is_sublist(labels, self.coco[image_id]["label_texts"]):
                self.coco.pop(image_id, None)

    def join(self, other):
        #joins coco dictionaries of two different datasets!
        #if A, B are datasets, this is AUB
        self.coco = {**self.coco, **other.coco}

    def random_image(self):
        #returns a random image from the dataset
        random_int = randint(0, len(self.coco))
        return self.coco[list(self.coco.keys())[random_int]]

    def find_uniques(self):
        #define unique variable
        self.unique_objects = set()
        #iterate through dataset to find unique objects
        for id in self.coco:
            self.unique_objects.update(set(self.coco[id]['label_texts']))

        #makes list from set and sorts alphabetically
        self.unique_objects = list(self.unique_objects)
        self.unique_objects.sort()

        #change to dictionary for a more helpful form...
        self.unique_objects = {object: num for num, object in enumerate(self.unique_objects)}

    def bag_of_words(self):
        #calculates bag of words embedding for each image in the dataset.
        for id in self.coco:
            self.coco[id]['bow'] = [0 for _ in self.unique_objects]
            for object in self.coco[id]['label_texts']:
                index = self.unique_objects[object]
                self.coco[id]['bow'][index] += 1

    def clean_urls(self):
        #cleans the data from images that simply dont have the url anymore!
        #note this takes quite a while if you have bad internet connection!
        for id in tqdm([*self.coco.keys()]):
            url = self.coco[id]['url']
            response = requests.get(url)
            if response.status_code != 200:
                del self.coco[id]

if __name__ == '__main__':

    import json

    # Load Json File with annotations
    with open("./annotations/instances_train2014.json", "rb") as f:
        data = json.load(f)

    #coco dataset...
    coco = COCO(data)
    coco2 = COCO(data)

    print (f"COCO keys: {coco.coco[list(coco.coco.keys())[0]].keys()}")

    #pruning dataset!!!

    print (f"Size of dataset before Pruning: {len(coco.coco)}")
    coco.prune_dataset(["surfboard"])
    print (f"Size of dataset after Pruning: {len(coco.coco)}")

    print (f"Size of dataset before Pruning: {len(coco2.coco)}")
    coco2.prune_dataset(["pizza"])
    print (f"Size of dataset after Pruning: {len(coco2.coco)}")

    #join
    coco.join(coco2)
    print (f"Size of dataset after Joining: {len(coco.coco)}")

    #example of a random image!
    print(coco.random_image())

    coco.find_uniques()
    coco.bag_of_words()

    print(coco.random_image())
