from see.Segment_Fitness import FitnessFunction
import copy
import numpy as np
from see.Segmentors import segmentor
from see.ColorSpace import colorspace
from see.Workflow import workflow
from see.Segment_Fitness import segment_fitness
from see import base_classes, GeneticSearch
from see import base_classes

def GeneratePairs(size):
    pairs=[]
    for i in range(size):
        for j in range(i+1,size):
            pairs.append([i,j])
    return pairs

def UncertaintyValue(segmenters,data):
    dataCopies=[copy.deepcopy(data) for i in range(len(segmenters))]
    segs=[workflow(i) for i in segmenters]
    dataCopies=[segs[i].runAlgo(dataCopies[i]) for i in range(len(segs))]
    uncertainty=([FitnessFunction(dataCopies[i[0]].mask,dataCopies[i[1]].mask) for i in GeneratePairs(len(dataCopies))])
    
    return np.mean(np.transpose(uncertainty)[0])

def ALSearch(segmenters,dataSet):
    uncertainties=[UncertaintyValue(segmenters,data) for data in dataSet]
    return copy.deepcopy(dataSet[np.argmin(uncertainties)])

def ActiveArgs(modelParams):
    return workflow.worklist[1]().algorithmspace[modelParams[3]]().paramindexes