from urllib.error import HTTPError
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
import numpy as np
import os, json

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


    norm_expl1 = nn.functional.normalize(explainer_1.float(), dim = 0)
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

    # has_error = True
    # while has_error:

    #     img = coco.random_image()
    #     img_url = img["url"]

    #     try:
    #         actual_image = read_image_from_url(img_url)
    #         has_error = False

    #     except HTTPError:
    #         pass


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
    actual_img = io.imread(img_url)

    lime_bbox_sim = []
    saliency_bbox_sim = []

    for box in img_bboxes:

        mask_tmp = (bbox_mask(actual_img.shape[:2], box))
 
        lime_bbox_sim.append(similarity(mask_tmp.flatten(), lime_tensor))

        saliency_bbox_sim.append(similarity(mask_tmp.flatten(), saliency_tensor))
        
    actual_img = transforms.ToTensor()(get_pil_transform()(im.fromarray(actual_img)))
    # img_boundry1 = mark_boundaries(actual_img.numpy(), mask_tmp.numpy().astype(int))  

    fig = plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(actual_img.permute(1, 2, 0))
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1, 2, 2)
    plt.imshow(mask_tmp, cmap="jet",alpha=0.8)
    plt.xticks([])
    plt.yticks([])
    plt.show()