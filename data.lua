require 'dataset-mnist'

nbTrainingPatches = 60000
nbTestingPatches = 10000

size = {32,32}
trainData = mnist.loadTrainSet(nbTrainingPatches, size)
testData = mnist.loadTrainSet(nbTrainingPatches, size)

return trainData, testData
