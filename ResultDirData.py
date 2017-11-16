#!/usr/bin/env python

import os
import numpy as np
from plotROCutils import readDescription

#----------------------------------------------------------------------

class ResultDirData:
    # keeps data which is common for the entire result directory
    def __init__(self, inputDir, useWeightsAfterPtEtaReweighting):
        self.inputDir = inputDir

        self.useWeightsAfterPtEtaReweighting = useWeightsAfterPtEtaReweighting

        self.description = readDescription(inputDir)

        # we don't have this for older trainings
        # self.trainWeightsBeforePtEtaReweighting = None

        # check for dedicated weights and labels file
        # train dataset
        fname = os.path.join(inputDir, "weights-labels-train.npz")

        if os.path.exists(fname):
            data = np.load(fname)
            self.origTrainWeights = data['origTrainWeights']
            # self.trainWeightsBeforePtEtaReweighting = data['weightBeforePtEtaReweighting']
            self.trainLabels = data['label']
        else:
            fname = os.path.join(inputDir, "weights-labels-train.npz.bz2")
            if os.path.exists(fname):
                import bz2
                data = np.load(bz2.BZ2File(fname))
                self.origTrainWeights = data['origTrainWeights']
                # self.trainWeightsBeforePtEtaReweighting = data['weightBeforePtEtaReweighting']
                self.trainLabels = data['label']
            else:
                # try the BDT file (but we don't have weights before eta/pt reweighting there)
                fname = os.path.join(inputDir, "roc-data-%s-mva.npz" % "train")
                data = np.load(fname)
                self.trainWeights = data['weight']
                self.trainLabels = data['label']
                self.trainWeightsBeforePtEtaReweighting = None
            
        #----------
        # test dataset
        #----------

        fname = os.path.join(inputDir, "weights-labels-test.npz")
        if os.path.exists(fname):
            data = np.load(fname)
            self.testWeights = data['weight']
            self.testLabels = data['label']

        else:
            fname = os.path.join(inputDir, "weights-labels-test.npz.bz2")

            if os.path.exists(fname):
                import bz2
                data = np.load(bz2.BZ2File(fname))
                self.testWeights = data['weight']
                self.testLabels = data['label']
            else:
                # try the BDT file
                fname = os.path.join(inputDir, "roc-data-%s-mva.npz" % "test")
                data = np.load(fname)
                self.testWeights = data['weight']
                self.testLabels = data['label']

    #----------------------------------------

    def getWeights(self, isTrain):
        if isTrain:

            # for training, returns the weights before eta/pt reweighting if available
            # self.trainWeightsBeforePtEtaReweighting is None does not work,
            # 
            # if there are no trainWeightsBeforePtEtaReweighting, these are array(None, dtype=object)

            
            # if self.useWeightsAfterPtEtaReweighting:
            #     assert self.hasTrainWeightsBeforePtEtaReweighting()
            #     return self.trainWeights
            # 
            # if self.trainWeightsBeforePtEtaReweighting.shape == (): 
            #     return self.trainWeights
            # else:
            #     return self.trainWeightsBeforePtEtaReweighting

            # original weights, before any reweighting
            return self.origTrainWeights

        else:
            return self.testWeights

    #----------------------------------------

    def getLabels(self, isTrain):
        if isTrain:
            return self.trainLabels
        else:
            return self.testLabels

    #----------------------------------------
            
    def hasTrainWeightsBeforePtEtaReweighting(self):
        return self.trainWeightsBeforePtEtaReweighting.shape != ()

#----------------------------------------------------------------------
