'''
This python script defines the neural net that will classify given url images.
'''

import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
from PIL import Image
from matplotlib import pyplot as plt
from skimage import io

def read_image_from_url(img_url):

    img = io.imread(img_url)
    img = Image.fromarray(img)
    return img

class Classifier:

    def __init__(self, arch):
        # th architecture to use
        self.arch = arch

        # load the pre-trained weights
        self.model_file = '%s_places365.pth.tar' % self.arch
        if not os.access(self.model_file, os.W_OK):
            weight_url = 'http://places2.csail.mit.edu/models_places365/' + self.model_file
            os.system('wget ' + weight_url)

        self.model = models.__dict__[arch](num_classes=365)
        checkpoint = torch.load(self.model_file, map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        self.model.load_state_dict(state_dict)
        self.model.eval()


        # load the image transformer
        self.centre_crop = trn.Compose([
                trn.Resize((256,256)),
                trn.CenterCrop(224),
                trn.ToTensor(),
                trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # load the class label
        file_name = 'categories_places365.txt'
        if not os.access(file_name, os.W_OK):
            synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
            os.system('wget ' + synset_url)

        self.classes = list()
        with open(file_name) as class_file:
            for line in class_file:
                self.classes.append(line.strip().split(' ')[0][3:])
        #contains classes
        self.classes = tuple(self.classes)


    # load the test image and classify it!
    def classify(self, img_url, top = 10):
      #loads test image from given url...
      img = io.imread(img_url)
      img = Image.fromarray(img)
      input_img = V(self.centre_crop(img).unsqueeze(0))

      # forward pass
      logit = self.model.forward(input_img)
      h_x = F.softmax(logit, 1).data.squeeze()
      probs, idx = h_x.sort(0, True)
      #top predictions!
      preds = []
      for i in range(0, top):
          preds.append([self.classes[idx[i]], float(probs[i])])
      return preds

    @staticmethod
    def get_preprocess_transform():
        normalize = trn.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])     
        transf = trn.Compose([
            trn.ToTensor(),
            normalize
        ])    

        return transf    

    # Normalize transformed image and to cast to Tensor    
    
    def batch_predict(self, images):

        preprocess_transform = self.get_preprocess_transform()
        batch = torch.stack(tuple(preprocess_transform(i) for i in images), dim=0)

        device = torch.device("cpu")
        self.model.to(device)
        batch = batch.to(device)
        
        logits = self.model(batch)
        probs = F.softmax(logits, dim=1)
        return probs.detach().cpu().numpy()



if __name__ == '__main__':

    predictor = Classifier("resnet18")

    image_url = "http://farm4.staticflickr.com/3153/2970773875_164f0c0b83_z.jpg"

    print (f"Image url: {image_url}")
    image = read_image_from_url(image_url)
    plt.imshow(image)
    print("Predictions")
    print(predictor.classify(image_url))
    plt.show()
