#!/usr/bin/env python

# given a results-* directory, finds the 
# .npz files with the network outputs
# for different epochs, calculates the ROC area
# and plots the progress

import sys, os

import glob, re
import numpy as np

from plotROCutils import addDirname, addNumEvents, readDescription
import plotROCutils

officialPhotonIdLabel = 'official photon id'

# benchmark for official photon id cut
officialPhotonIdCut = 0.23

from ResultDirData import ResultDirData

#----------------------------------------------------------------------

def drawSingleROCcurve(resultDirRocs, epoch, isTrain, label, color, lineStyle, linewidth, label_args = {}):

    baseDir = os.path.basename(os.path.normpath(resultDirRocs.getInputDir()))

    auc, numEvents, fpr, tpr, thresholds = resultDirRocs.getFullROCcurve(epoch, isTrain)

    # TODO: we could add the area to the legend
    pylab.plot(fpr, tpr, lineStyle, color = color, linewidth = linewidth, 
               label = label.format(auc = auc, baseDir = baseDir, epoch = epoch, **label_args))

    return fpr, tpr, numEvents

#----------------------------------------------------------------------

def drawBenchmarkPoints(resultDirRocs, epoch, isTrain, color, benchmarkPoints):
    # draw benchmark points on single roc curves
    # for the reference sample and 
    # 
    # @param benchmarkPoints is a list of cuts on the BDT (official)
    # photon ID

    auc, numEvents, fpr,    tpr,    thresholds    = resultDirRocs.getFullROCcurve(epoch, isTrain)

    auc, numEvents, fprBDT, tprBDT, thresholdsBDT = resultDirRocs.getFullROCcurve("BDT", isTrain)

    for benchmarkPoint in benchmarkPoints:

            import bisect

            # note that we use only thresholdsBDT to find the indices
            # then we get the BDT fpr (background efficiency)
            # and find the corresponding index for our training
            # (we can't reuse the indices from the BDT training)

            # note that thresholds is in decreasing order hence
            # the left/right crossing
            wpIndexRight = bisect.bisect_left(thresholdsBDT[::-1], benchmarkPoint)
            wpIndexLeft  = bisect.bisect_right(thresholdsBDT[::-1], benchmarkPoint)

            # assume the working point is away from the border
            wpFPRbdt = 0.5 * (fprBDT[::-1][wpIndexRight] + fprBDT[::-1][wpIndexLeft])
            wpTPRbdt = 0.5 * (tprBDT[::-1][wpIndexRight] + tprBDT[::-1][wpIndexLeft])

            # now get the TPR corresponding to wpFPRbdt for our training
            import scipy
            wpTPR    = scipy.interpolate.interp1d(fpr[::-1], tpr[::-1])(wpFPRbdt)

            # print "thresholds at working point: left=",thresholdsBDT[::-1][wpIndexLeft],"right=",thresholdsBDT[::-1][wpIndexRight], "bg eff=",wpFPRbdt, "sig eff bdt=",wpTPRbdt, "sig eff=",wpTPR

            pylab.plot([wpFPRbdt], [wpTPRbdt], 'o', color = color)

            xlim = pylab.xlim()
            textOffset = 0.1 * (xlim[1] - xlim[0])

            pylab.text(wpFPRbdt + textOffset, wpTPRbdt, '%.1f %%' % (100*wpTPRbdt), va = 'center')


            pylab.plot([wpFPRbdt], [wpTPR],    'o', color = color)
            # pylab.gca().annotate('%.1f%%' % (100*wpTPR),    (wpFPRbdt, wpTPR))
            pylab.text(wpFPRbdt + textOffset, wpTPR, '%.1f%%' % (100*wpTPR), va = 'center')

    

#----------------------------------------------------------------------

def plotGradientMagnitudes(inputDir, mode):

    # @param mode can be 
    #    'stat'   plot the median/mean and error bars
    #    'detail' plot the actual gradient magnitude values
    #
    # @return True if something was plotted

    inputFiles = glob.glob(os.path.join(inputDir, "gradient-magnitudes-*.npz"))
    
    # maps from epoch number to gradient magnitudes
    epochToGradientMagnitudes = {}

    # sort by epoch number
    for inputFname in inputFiles:

        basename = os.path.basename(inputFname)

        # example names:
        #  roc-data-test-mva.t7
        #  roc-data-train-0002.t7

        mo = re.match("gradient-magnitudes-(\d+).npz$", basename)
        if not mo:
            print >> sys.stderr,"warning: skipping",inputFname
            continue
        epoch = int(mo.group(1))
        
        thisData = np.load(inputFname)['gradientMagnitudes']

        assert not epoch in epochToGradientMagnitudes

        epochToGradientMagnitudes[epoch] = thisData

    # end of loop over input files

    if not epochToGradientMagnitudes:
        return False

    pylab.figure(facecolor='white')

    # median number of gradient evaluations per epoch
    # (for filling in missing ones)
    # normally they should all be the same
    medianLength = np.median(
        [ len(x) for x in epochToGradientMagnitudes.values() ]
        )

    epochs = sorted(epochToGradientMagnitudes.keys())


    if mode == 'detail':
        # plot epoch by epoch with different colors
        for epoch in epochs:
            yvalues = epochToGradientMagnitudes[epoch]

            # choose x axis normalization such that each epoch
            # corresponds to an interval of one
            xvalues = np.linspace(epoch, epoch + 1,
                                  num = len(yvalues),
                                  endpoint = False)

            pylab.plot(xvalues, yvalues)

    elif mode == 'stat':
        # plot mean and standard deviations
        xvalues = epochs

        yvalues = [ np.mean(epochToGradientMagnitudes[epoch]) for epoch in epochs ]
        yerrs   = [ np.std(epochToGradientMagnitudes[epoch]) for epoch in epochs ]

        pylab.errorbar(xvalues, yvalues, yerr = yerrs)

    else:
        raise Exception("unsupported mode " + mode)

    pylab.xlabel('epoch')
    pylab.ylabel('gradient magnitude')

    pylab.grid()

    # minor ticks to indicate start of epochs
    ax = pylab.gca()
    ax.set_xticks(epochs, minor = True)
    ax.grid(which='minor', alpha=0.2)                                                




    return True

#----------------------------------------------------------------------

def updateHighestTPR(highestTPRs, fpr, tpr, maxfpr):
    if maxfpr == None:
        return

    # find highest TPR for which the FPR is <= maxfpr
    highestTPR = max([ thisTPR for thisTPR, thisFPR in zip(tpr, fpr) if thisFPR <= maxfpr])
    highestTPRs.append(highestTPR)

#----------------------------------------------------------------------
def drawLast(resultDirRocs, xmax = None, ignoreTrain = False,
             savePlots = False,
             legendLocation = None,
             addTimestamp = True,
             refResultDirRocs = None,             
             ):
    # plot ROC curve for last epoch only

    # @param refResultDirRocs if not None, plot agains this reference data
    # instead of the BDT value

    hasRef = refResultDirRocs is not None

    pylab.figure(facecolor='white')
    
    #----------

    # find the highest epoch for which both
    # train and test samples are available

    epochNumber = resultDirRocs.findLastCompleteEpoch(ignoreTrain)

    if hasRef:
        refEpochNumber = refResultDirRocs.findLastCompleteEpoch(ignoreTrain)
        assert refEpochNumber is not None
    else:
        refEpochNumber = None

    #----------
    if hasRef:
        labelTemplate = "{baseDir} {sample} (last auc={auc:.3f}, epochs={epoch})"
        labelTemplateRef = labelTemplate
    else:
        labelTemplate = "NN ({sample} auc {auc:.3f})"
        labelTemplateRef = officialPhotonIdLabel + " ({sample} auc {auc:.3f})"


    #----------

    highestTPRs = []
    #----------

    # maps from sample type top number of events
    numEvents = {}

    for sample, color in (
        ('train', 'blue'),
        ('test', 'red'),
        ):

        isTrain = sample == 'train'

        if ignoreTrain and isTrain:
            continue

        #----------
        # plot directory of interest
        #----------

        # take the last epoch
        if epochNumber != None:
            fpr, tpr, numEvents[sample] = drawSingleROCcurve(resultDirRocs, epochNumber, isTrain, labelTemplate, color, '-', 2, label_args = dict(sample = sample))
            updateHighestTPR(highestTPRs, fpr, tpr, xmax)

        #----------
        # draw reference
        #----------

        if hasRef:
            # plot reference curve for comparison
            fpr, tpr, numEvents[sample] = drawSingleROCcurve(refResultDirRocs, refEpochNumber, isTrain, labelTemplateRef, color, '--', 2, label_args = dict(sample = sample))
            updateHighestTPR(highestTPRs, fpr, tpr, xmax)

        else:
            # draw the ROC curve for the MVA id if available
            if resultDirRocs.hasBDTroc(isTrain):
                fpr, tpr, dummy = drawSingleROCcurve(resultDirRocs, 'BDT', isTrain, labelTemplateRef, color, '--', 1, label_args = dict(sample = sample))
                updateHighestTPR(highestTPRs, fpr, tpr, xmax)            

                # draw comparison benchmark points for test sample
                # TODO: we should also support this for arbitrary references
                if not isTrain:
                    drawBenchmarkPoints(resultDirRocs, epochNumber, isTrain, color, benchmarkPoints = [ officialPhotonIdCut ])

            

        # draw another reference curve if specified
        # TODO: generalize this to multiple references where the BDT
        #       output is the default one

    pylab.xlabel('fraction of false positives')
    pylab.ylabel('fraction of true positives')

    if xmax != None:
        pylab.xlim(xmax = xmax)
        # adjust y scale
        pylab.ylim(ymax = 1.1 * max(highestTPRs))

    pylab.grid()
    pylab.legend(loc = legendLocation)

    inputDir = resultDirRocs.getInputDir()

    if addTimestamp:
        plotROCutils.addTimestamp(inputDir)

    addDirname(inputDir)
    addNumEvents(numEvents.get('train', None), numEvents.get('test', None))

    inputDirDescription = resultDirRocs.getInputDirDescription()
    if inputDirDescription != None:

        title = str(inputDirDescription)

        if epochNumber != None:
            title += " (epoch %d)" % epochNumber

        pylab.title(title)

    if savePlots:
        for suffix in (".png", ".pdf", ".svg"):
            outputFname = os.path.join(inputDir, "last-auc")

            if xmax != None:
                outputFname += "-%.2f" % xmax

            outputFname += suffix

            pylab.savefig(outputFname)
            print "saved figure to",outputFname

#----------------------------------------------------------------------

def plotAucEvolution(resultDirData, resultDirRocs,
                     refResultDirData = None, refResultDirRocs = None,
                     ignoreTrain = False,
                     legendLocation = None,
                     nodate = False,
                     savePlots = False):
    # plots the evolution of the ROCs vs. epoch
    #
    # if refResultDirData and refResultDirRocs are not None,
    # plots these instead of the BDT/MVA as reference

    mvaROC, rocValues = resultDirRocs.getAllROCs()

    pylab.figure(facecolor='white')

    hasRef = refResultDirData is not None

    for sample, color in (
        ('train', 'blue'),
        ('test', 'red'),
        ):

        if ignoreTrain and sample == 'train':
            continue

        #----------
        def plotEvolution(rocValues, style, label, inputDir):

            # sorted by ascending epoch
            epochs = sorted(rocValues[sample].keys())
            aucs = [ rocValues[sample][epoch] for epoch in epochs ]

            pylab.plot(epochs, aucs, style, 
                       label = label.format(
                         auc = aucs[-1], 
                         maxEpoch = max(epochs),
                         baseDir = os.path.basename(os.path.normpath(inputDir)),
                    ), color = color, linewidth = 2)
            return epochs

        #----------

        if hasRef:
            # plot comparison
            plotEvolution(rocValues, '-o', "{baseDir} " + sample + " (last auc={auc:.3f}, epochs={maxEpoch})", resultDirData.inputDir)
            plotEvolution(refResultDirRocs.getAllROCs()[1], '--', "{baseDir} " + sample + " (last auc={auc:.3f}, epochs={maxEpoch})", refResultDirData.inputDir)

        else:
            # plot single training vs. MVA/BDT
            epochs = plotEvolution(rocValues, '-o', "NN " + sample + " (last auc={auc:.3f})", resultDirData.inputDir)
        
            # draw a line for the MVA id ROC if available
            auc = mvaROC[sample]
            if auc != None:
                pylab.plot( pylab.gca().get_xlim(), [ auc, auc ], '--', color = color,
                            label = "%s (%s auc=%.3f)" % (officialPhotonIdLabel, sample, auc))

    pylab.grid()

    if hasRef:
        pylab.xlabel('training epoch')
    else:
        pylab.xlabel('training epoch (last: %d)' % max(epochs))
    pylab.ylabel('AUC')

    pylab.legend(loc = legendLocation)

    #----------

    if not hasRef:
        if resultDirData.description != None:
            pylab.title(resultDirData.description)

        if not nodate:
            plotROCutils.addTimestamp(resultDirData.inputDir)

        addDirname(resultDirData.inputDir)

    #----------

    if savePlots:
        for suffix in (".png", ".pdf", ".svg"):
            # for comparisons the plot will go to the 
            # non-ref output directory

            outputFname = "auc-evolution"
            if hasRef:
                outputFname += "-comparison"
            
            outputFname = os.path.join(inputDir, outputFname + suffix)
            pylab.savefig(outputFname)
            print "saved figure to",outputFname


#----------------------------------------------------------------------
# main
#----------------------------------------------------------------------
if __name__ == '__main__':

    from optparse import OptionParser
    parser = OptionParser("""

      usage: %prog [options] result-directory

    """
    )

    parser.add_option("--last",
                      default = False,
                      action="store_true",
                      help="plot ROC curve for last epoch only",
                      )

    parser.add_option("--both",
                      default = False,
                      action="store_true",
                      help="plot AUC evolution and last AUC curve",
                      )

    parser.add_option("--ignore-train",
                      dest = 'ignoreTrain',
                      default = False,
                      action="store_true",
                      help="do not look at train values",
                      )

    parser.add_option("--min-epoch",
                      dest = 'minEpoch',
                      type = int,
                      default = None,
                      help="first epoch to plot (useful e.g. if the training was far off at the beginning)",
                      )

    parser.add_option("--max-epoch",
                      dest = 'maxEpoch',
                      type = int,
                      default = None,
                      help="last epoch to plot (useful e.g. if the training diverges at some point)",
                      )

    parser.add_option("--exclude-epochs",
                      dest = 'excludedEpochs',
                      type = str,
                      default = None,
                      help="comma separated list of epochs to ignore (e.g. in case of problematic/missing output files)",
                      )

    parser.add_option("--save-plots",
                      dest = 'savePlots',
                      default = False,
                      action="store_true",
                      help="save plots in input directory",
                      )

    parser.add_option("--legend-loc",
                      dest = 'legendLocation',
                      default = 'lower right',
                      help="location of legend in plots",
                      )

    parser.add_option("--weights-after-pt-eta-reweighting",
                      dest = 'useWeightsAfterPtEtaReweighting',
                      default = False,
                      action = "store_true",
                      help="use weights (for training) after pt/eta reweighting",
                      )


    parser.add_option("--nodate",
                      default = False,
                      action = 'store_true',
                      help="do not add the timestamp to plots",
                      )

    parser.add_option("--refdir",
                      default = None,
                      help="reference directory to compare to (instead of BDT)",
                      )

    (options, ARGV) = parser.parse_args()

    assert len(ARGV) == 1, "usage: plotROCs.py result-directory"

    inputDir = ARGV.pop(0)

    if options.excludedEpochs != None:
        options.excludedEpochs = [ int(x) for x in options.excludedEpochs.split(',') ]

    #----------

    resultDirData = ResultDirData(inputDir, options.useWeightsAfterPtEtaReweighting)

    import pylab

    from ResultDirRocs import ResultDirRocs
    resultDirRocs = ResultDirRocs(resultDirData,
                                  minEpoch = options.minEpoch,
                                  maxEpoch = options.maxEpoch,
                                  excludedEpochs = options.excludedEpochs)

    #----------
    # get information from reference directory
    #----------
    if options.refdir is not None:
        refResultDirData = ResultDirData(options.refdir, options.useWeightsAfterPtEtaReweighting)
        refResultDirRocs = ResultDirRocs(refResultDirData,
                                         minEpoch = options.minEpoch,
                                         maxEpoch = options.maxEpoch,
                                         excludedEpochs = options.excludedEpochs)
    else:
        refResultDirData = None
        refResultDirRocs = None

    #----------

    if options.last or options.both:

        drawLast(resultDirRocs, ignoreTrain = options.ignoreTrain,
                 savePlots = options.savePlots,
                 legendLocation = options.legendLocation,
                 addTimestamp = not options.nodate,
                 refResultDirRocs = refResultDirRocs)

        # zoomed version
        # autoscaling in y with x axis range manually
        # set seems not to work, so we implement
        # something ourselves..
        drawLast(resultDirRocs, xmax = 0.05, ignoreTrain = options.ignoreTrain,
                 savePlots = options.savePlots,
                 legendLocation = options.legendLocation,
                 addTimestamp = not options.nodate,
                 refResultDirRocs = refResultDirRocs)


    if not options.last or options.both:
        #----------
        # plot evolution of area under ROC curve vs. epoch
        #----------
        print "plotting AUC evolution"

        plotAucEvolution(
            resultDirData,
            resultDirRocs,
            ignoreTrain = options.ignoreTrain,
            legendLocation = options.legendLocation,
            nodate = options.nodate,
            savePlots = options.savePlots,

            refResultDirData = refResultDirData, 
            refResultDirRocs = refResultDirRocs,
            )

    #----------

    if not options.last or options.both:

        #----------
        # plot correlation of train and test AUC
        #----------

        import plotAUCcorr

        plotAUCcorr.doPlot(resultDirRocs, addTimestamp = not options.nodate)

        if options.savePlots:
            for suffix in (".png", ".pdf", ".svg"):
                outputFname = os.path.join(inputDir, "auc-corr" + suffix)
                pylab.savefig(outputFname)
                print "saved figure to",outputFname


        #----------
        # plot gradient magnitudes
        #----------

        plotted = plotGradientMagnitudes(inputDir, mode = 'detail')

        if plotted and options.savePlots:
            for suffix in (".png", ".pdf", ".svg"):
                outputFname = os.path.join(inputDir, "gradient-magnitude" + suffix)
                pylab.savefig(outputFname)
                print "saved figure to",outputFname


    #----------

    if not options.savePlots:
        # show plots interactively
        pylab.show()


