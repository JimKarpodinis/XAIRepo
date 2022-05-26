'''
Utility of this file is to transform a coco json file to an extracted json file.
Said file will simply contain a said image url and annotations, with keys it's coco id.

This file filters two special words named special_word1, special_word2. Then creates the json file
containing images only that include annotations of one or both categories in them.

Used as: python json_extractor.py input_json output_json
'''

import json
import sys

input_json = sys.argv[1]
output_json = sys.argv[2]

special_word1 = 'pizza'
special_word2 = 'surfboard'

with open(input_json, 'r') as json_file:
    #load starting json file...
    data = json.load(json_file)
    #making images, annotations, categories ...
    images = data['images']
    annotations = data['annotations']
    categories = data['categories']
    #this is the basic dictionary...
    image_dict = dict()
    for img in images:
        image_id = img['id']
        image_dict[image_id] = dict()
        #setting url of picture
        image_dict[image_id]['url'] = img['coco_url']
        #setting empty annotation set for later...
        image_dict[image_id]['annotations'] = set()

    #helper dictionary to transform a category id to its name...
    category_dict = dict()
    for category in categories:
        category_dict[category['id']] = category['name']

    #now the extraction of all categories per image will happen...
    for annotation in annotations:
        image_id = annotation['image_id']
        category_id = annotation['category_id']
        img = image_dict[image_id]
        #get name of category
        name = category_dict[category_id]
        #append category to the image annotation set...
        image_dict[image_id]['annotations'].add(name)

del images, annotations, categories

#filtering images not containing either pizza or surfboard...
for id in list(image_dict.keys()):

    l1 = special_word1 in image_dict[id]['annotations']
    l2 = special_word2 in image_dict[id]['annotations']

    #TypeError: Object of type set is not JSON serializable
    image_dict[id]['annotations'] = list(image_dict[id]['annotations'])

    #delete all images not containing one of two special searches!
    if not l1 and not l2:
        del image_dict[id]

#saving extracted json file...
json_string = json.dumps(image_dict)
with open(output_json, 'w') as output:
    output.write(json_string)
