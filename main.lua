require 'torch'
require 'cutorch'
require 'nn'

function init(opt)
    torch.setdefaulttensortype('torch.FloatTensor')
    torch.manualSeed(opt.seed)
    cutorch.manualSeedAll(opt.seed)
end

function model_init(model, criterion, opt)
    print(sys.COLORS.red ..  '==> use GPU')
    require 'cutorch'
    require 'cunn'
    require 'cudnn'
    cutorch.setDevice(opt.devid)
    local devid = cutorch.getDevice()
    local gpu_info = cutorch.getDeviceProperties(devid)
    print(sys.COLORS.red ..  '==> using GPU #' .. devid .. ' ' .. gpu_info.name .. ' (' .. math.ceil(gpu_info.totalGlobalMem/1024/1024/1024) ..  'GB)')
    model:cuda()
    criterion:cuda()
end

function main()
    local lapp = require 'pl.lapp'
    local opt = lapp[[
    -r,--learningRate        (default 0.5)         learning rate
    -d,--learningRateDecay   (default 1e-7)        learning rate decay (in # samples)
    -w,--weightDecay         (default 1e-5)        weight decay
    -m,--momentum            (default 0.1)         momentum
    -b,--backend             (default cunn)        nn or cunn or cudnn
    -i,--devid               (default 1)           device ID (if using CUDA; NOTE: device ID starts from 1)
    -n,--nThreads            (default 2)           number of  data loading threads
    -b,--batchSize           (default 64)          batch size
    -e,--eEpoch              (default 50)          number of total epochs (-1 means infinity)
    -z,--sEpoch              (default 1)           number of start epoch
    -l,--load                (default none)        load model from specified file
    -s,--seed                (default 1)           seed of random
    -o,--out                 (default results)     save directory
    -c,--confusion           (default true)        output confusion matrix for each epoch
    -q,--dataset                (default mnist)       dataset name
    -p,--data                   (default data/mnist.hdf5) data location
    ]]


    local DataLoader = require 'dataloader'
    local Trainer = require 'train'
    local models = require 'model'
    local model = models.model
    local criterion = models.criterion

    init(opt)
    model_init(model, criterion, opt)

    opt.classes = {'0','1','2','3','4','5','6','7','8','9'} -- for confusion matrix
    -- opt.classes = {'A','B','C','C','D','E','F','G','H','I'} -- for confusion matrix
    print("setup dataloader")
    local trainLoader, valLoader = DataLoader.create(opt)
    print("setup trainer")
    local trainer = Trainer(model, criterion, opt)
    local epoch = opt.sEpoch
    print("start training")
    while true do
        if opt.eEpoch > 0 and opt.eEpoch < epoch then
            break
        end

        trainer:train(epoch, trainLoader)
        trainer:test(epoch, valLoader)

        epoch = epoch + 1
    end
end

main()
