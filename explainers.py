from urllib.error import HTTPError
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
import numpy as np
import os, json
import matplotlib.patches as patches
import torch
from torchvision import models, transforms
from torch.autograd import Variable
import torch.nn.functional as F
from predictor import Classifier, read_image_from_url
from lime import lime_image
from skimage import io
from skimage.segmentation import mark_boundaries
from coco_dataset import COCO
from PIL import Image as im
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Import dependencies

def get_pil_transform():
    transf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224)
    ])

    return transf

    # Transform PIL image

def lime_explainer(img, clf, pill_transf = get_pil_transform()):
    explainer = lime_image.LimeImageExplainer()

    explanation = explainer.explain_instance(np.array(pill_transf(img)),
                                            clf.batch_predict, # classification function
                                            top_labels=5,
                                            hide_color=0,
                                            num_samples=1000) # number of images that will be sent to classification function


    return explanation

def saliency_explainer(img, clf, pill_transf=get_pil_transform()):

    #we don't need gradients w.r.t. weights for a trained model
    for param in clf.model.parameters():
        param.requires_grad = False

    #transoform input PIL image to torch.Tensor and normalize
    input = transforms.ToTensor()(pill_transf(img))
    input.unsqueeze_(0)

    #we want to calculate gradient of higest score w.r.t. input
    #so set requires_grad to True for input
    input.requires_grad = True
    #forward pass to calculate predictions
    preds = clf.model.forward(input)
    score, _ = torch.max(preds, 1)
    #backward pass to get gradients of score predicted class w.r.t. input image
    score.backward()
    #get max along channel axis
    slc, _ = torch.max(torch.abs(input.grad[0]), dim=0)
    #normalize to [0..1]
    slc = (slc - slc.min())/(slc.max()-slc.min())

    return slc

def visualize_saliency(slc, img, pill_transf=get_pil_transform()):

    #plot image and its saleincy map
    fig = plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(img.resize((256, 256)))
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1, 2, 2)
    plt.imshow(slc.numpy(), cmap="jet",alpha=0.8)
    plt.xticks([])
    plt.yticks([])
    plt.show()

def similarity(explainer_1, explainer_2):


    norm_expl1 = nn.functional.normalize(explainer_1.float(), dim=0)
    norm_expl2 = nn.functional.normalize(explainer_2.float(), dim=0)

    return norm_expl1 @ norm_expl2

def bbox_mask(img_shape, bbox, transform=get_pil_transform()):

        mask = np.zeros(img_shape, dtype=np.uint8)
        x_min, y_min, w, h = np.round(bbox).astype(int)

        y_max = h + y_min

        x_max = w + x_min

        mask[y_min:y_max, x_min:x_max] = 1

        img_tmp = im.fromarray(mask)
        output = transforms.ToTensor()(transform(img_tmp))

        return output.squeeze(0)

def visualize_bboxes(image, bboxes, figsize=(5, 5)):
    #for bbox visualization...

    fig = plt.figure(figsize=figsize)

    ax = fig.add_subplot(1,1,1)

    plt.xticks([])
    plt.yticks([])

    for box in bboxes:
        bb = patches.Rectangle((box[0],box[1]), box[2],box[3], linewidth=2, edgecolor="blue", facecolor="none")
        ax.add_patch(bb)

    ax.imshow(image)

    return fig

def prepare_data(dataset, labels, threshold = 60, include_other = False):
    #This function prepares data, to be explained with trees!
    #receives coco dataset and corresponding labeled data!

    #features, containing bow embeddings!
    X = list()
    #labels in integer form, to be used in training
    y = list()
    #label images!
    #keeps frequencies, so as to remove infrequent labels!!!
    freqs = dict()

    for id in tqdm(dataset.coco):
        try:
            label = labels[str(id)]
            X.append(dataset.coco[id]['bow'])
            y.append(label)

            if label not in freqs:
                freqs[label] = 1
            else:
                freqs[label] += 1
        except:
            pass

    start_length = len(y)

    #filter infrequent labels!!!
    X_ = list()
    y_ = list()
    if threshold != -1:
        for label, features in tqdm(zip(y, X), desc="filter infrequent classes"):
            if freqs[label] >= threshold:
                #keep these only!
                X_.append(features)
                y_.append(label)
            else:
                if include_other:
                    #all low frequency classes are bumped to one...
                    X_.append(features)
                    y_.append('other')
    X = X_
    y = y_

    finish_length = len(y)

    del X_, y_

    print('Percentage of filtered variables: ', (1-(finish_length/start_length))*100,"%")

    #finds an integer for each unique label!
    y_unique = list(set(y))
    y_unique.sort()
    y_unique = {label: num for num, label in enumerate(y_unique)}

    for i in range(len(y)):
        y[i] = y_unique[y[i]]

    return X, y, y_unique


def tree_explainer(dataset, labels, threshold = 60, include_other = False,
                                                test_size=0.33, random_state=42,
                                                                    **tree_kwargs):
    '''
    This explainer uses a tree classifier to produce an explanation
    about the predicted labels of a coco classifier!

    dataset: coco object
    labels: dictionary containing predicted labels from a black box predictor!
    threshold:

    tree_kwargs: keyword arguments for an sklearn tree classifier!!!
    '''

    #preparing data for training the tree!!!
    X, y, y_unique = prepare_data(dataset, labels,
                    threshold = threshold, include_other = include_other)

    #splitting data train/test...
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                test_size=test_size, random_state=random_state)

    #define tree...
    tree_ = DecisionTreeClassifier(**tree_kwargs)

    tree_.fit(X_train, y_train)

    print('Traing accuracy: ',tree_.score(X_train, y_train))
    print('Testing accuracy: ',tree_.score(X_test, y_test))

    return tree_, y_unique


if __name__ == "__main__":

    import json

    # Load Json File with annotations
    with open("./annotations/instances_train2014.json", "rb") as f:
        data = json.load(f)

    #coco dataset...
    coco = COCO(data)
    coco2 = COCO(data)

    #pruning dataset!!!

    coco.prune_dataset(["surfboard"])
    coco2.prune_dataset(["pizza"])

    #join
    coco.join(coco2)

    img_url = coco.coco[327758]["url"]
    actual_image = read_image_from_url(img_url)

    clf = Classifier("resnet18")
    explanation_lime = lime_explainer(actual_image, clf)
    print(clf.classify(img_url)[0][0])

    temp, mask = explanation_lime.get_image_and_mask(explanation_lime.top_labels[0], positive_only=True, num_features=10, hide_rest=False)
    img_boundry1 = mark_boundaries(temp/255.0, mask, outline_color = [250.0, 20.0, 5.0])

    lime_tensor = torch.from_numpy(mask.flatten())

    plt.imshow(img_boundry1)
    plt.show()

    saliency = saliency_explainer(actual_image, clf)

    saliency_tensor = saliency.flatten()

    explainer_similarity = similarity(lime_tensor, saliency_tensor)

    print(f"Explainer similarity: {explainer_similarity}")

    visualize_saliency(saliency, actual_image)

    temp_img = coco.coco[327758]

    img_url = temp_img["url"]
    img_bboxes = temp_img["bboxes"]
    # actual_img = io.imread(img_url)

    lime_bbox_sim = []
    saliency_bbox_sim = []

    for box in img_bboxes:

        mask_tmp = (bbox_mask(actual_image.size[::-1], box))

        lime_bbox_sim.append(similarity(mask_tmp.flatten(), lime_tensor))

        saliency_bbox_sim.append(similarity(mask_tmp.flatten(), saliency_tensor))

    print(coco.coco[327758]['label_texts'])

    print(torch.Tensor(lime_bbox_sim)/torch.sum(torch.Tensor(lime_bbox_sim)))

    print(torch.Tensor(saliency_bbox_sim)/torch.sum(torch.Tensor(saliency_bbox_sim)))

    #tree explainer...
    #calculates bag of words!!!
    coco.find_uniques()
    coco.bag_of_words()

    with open("./saved_labels/labels.json", "r") as f:
        labels = json.load(f)

    tree_, unique_labels = tree_explainer(coco, labels, max_depth = 3)

    import re

    print(re.sub('class: ([0-9]+)',
            lambda x: 'Class: '+[key for key in unique_labels if int(x.groups(1)[0]) == unique_labels[key]][0],
            tree.export_text(tree_,
            feature_names=[name for name in coco.unique_objects])))


    fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=800)

    tree.export_graphviz(tree_,
                   out_file = './tree.dot',
                   feature_names = [name for name in coco.unique_objects],
                   class_names=[name for name in unique_labels],
                   filled = True)

    os.system('dot -Tpng tree.dot -o tree.png')
