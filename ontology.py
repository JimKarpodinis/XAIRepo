from owlready2 import *
from nltk.corpus import wordnet as wn
import nltk
from rdflib.graph import Graph
import networkx as nx
import matplotlib.pyplot as plt
from rdflib.extras.external_graph_libs import rdflib_to_networkx_multidigraph
import io
import pydotplus
from IPython.display import display, Image
from rdflib.tools.rdf2dot import rdf2dot
from tqdm import tqdm

class Ontology:
    def __init__(self, iri, load = False):
        if not load:
            #the ontology is defined by a given iri
            self.iri = iri
            self.onto = get_ontology(iri)

            #here we define the main attributes and roles.
            with self.onto:
                class Image(Thing):
                    pass

                class DepictedObject(Thing):
                    pass

                class hasObject(Image>>DepictedObject):
                    pass
        else:
            #if we want to load, it needs for an iri the file directory it was stored.
            self.iri = iri
            self.onto = get_ontology(iri).load()

    def image_ontology(self, label_texts, id):
        #given the annotations an image contains and its id,
        #it produces the image ontology, using NLTK wordnet.

        #creates an object, named with the id, of type image.
        im = self.onto['Image'](str(id))

        for obj_name in label_texts:
            #iterates through all annotated objects in the image

            #produces the synsets of wordnet for said object
            synsets = wn.synsets(obj_name)

            #if no synsets, no processing...
            if len(synsets)<1:
                continue

            #creates an object of type DepictedObject
            obj = self.onto['DepictedObject']()
            #connects image object with said DepictedObject using role hasObject
            im.hasObject.append(obj)
            #we only use the first synset
            synset = wn.synsets(obj_name)[0].name()
            #actual object depicted like surfboard, pizza, etc ...
            actual_object = synset.split(".")[0]

            if self.onto[actual_object] in self.onto.classes()
                #if actual_object class has been created,
                #connect it with the depicted object class using the role is_a
                obj.is_a.append(self.onto[actual_object])
            else:
                #if there is no actual_object class (like pizza class), create the wordnet hierarchy!
                hyper = lambda s:s.hypernyms()
                hypers = [s.name().split(".")[0] for s in list(wn.synset(synset).closure(hyper))]
                #hypers contains the wordnet hierarchy starting with the root Thing
                #and ending with the actual object...
                hypers = reversed(hypers)
                father = Thing
                for h in hypers:
                    #for each wordnet object, if it doesnt exist create it.
                    if self.onto[h] not in self.onto.classes():
                        with self.onto:
                            cl = types.new_class(h,(father,))
                    father = self.onto[h]
                if self.onto[actual_object] not in self.onto.classes():
                    with self.onto:
                        cl = types.new_class(actual_object,(father,))
                #again we connect actual_object class with DepictedObject class using the role is_a
                with self.onto:
                    obj.is_a.append(self.onto[actual_object])

    def dataset_ontology(self, dataset):
        #Simple iteration of all images stored in dataset dictionary!
        for id in tqdm(dataset.coco):
            #for each image id...
            self.image_ontology(dataset.coco[id]['label_texts'], id)

    def visualize(self, file, format = "nt", write_pdf=True, pdf="onto.pdf"):
        #visualize the ontology, either in a notebook or in a pdf format.
        g = Graph()
        result = g.parse(file, format = format)
        stream = io.StringIO()
        rdf2dot(g, stream, opts = {display})
        dg = pydotplus.graph_from_dot_data(stream.getvalue())
        if write_pdf:
            dg.write_pdf(pdf)
        else:
            png = dg.create_png()
            display(Image(png))


    def save(self, file, format='ntriples'):
        self.onto.save(file, format=format)

if __name__ == '__main__':

    ontology = Ontology('http://myontology2/')

    from coco_dataset import COCO
    import json

    # Load Json File with annotations
    with open("./annotations/instances_train2014.json", "rb") as f:
        data = json.load(f)

    #coco dataset...
    coco = COCO(data)

    coco.prune_dataset(["surfboard"])

    print(coco.coco[327758])

    id = 327758
    label_texts = coco.coco[id]['label_texts']

    ontology.image_ontology(label_texts, id)

    #ontology.dataset_ontology(coco)

    with ontology.onto:
        class Sea(Thing):
            equivalent_to = [ontology.onto["hasObject"].only(ontology.onto["surfboard"])]

    ontology.save('onto.nt',format='ntriples')

    ontology.visualize("./onto.nt", format = "nt", pdf = "onto.pdf")

    # ontology2 = Ontology("./onto.nt", load = True)
