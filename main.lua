require 'torch'
require 'cutorch'
require 'nn'

function init(opt)
    torch.setdefaulttensortype('torch.FloatTensor')
    torch.manualSeed(opt.seed)
    cutorch.manualSeedAll(opt.seed)
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
    -n,--threads             (default 8)           number of threads
    -b,--batchSize           (default 64)          batch size
    -e,--eEpoch              (default 50)          number of total epochs (-1 means infinity)
    -z,--sEpoch              (default 1)           number of start epoch
    -l,--load                (default none)        load model from specified file
    -s,--seed                (default 1)           seed of random
    -o,--out                 (default results)     save directory
    -c,--confusion           (default true)        output confusion matrix for each epoch
    ]]

    init(opt)

    local trainData, valData = require 'data'
    local Trainer = require 'train'
    local models = require 'model'
    local model = models.model
    local criterion = models.criterion

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

    opt.classes = {'0','1','2','3','4','5','6','7','8','9'}
    local trainer = Trainer(model, criterion, opt, trainData, valData)
    local epoch = opt.sEpoch
    while true do
        if opt.eEpoch > 0 and opt.eEpoch < epoch then
            break
        end

        trainer:train(epoch)
        -- trainer:test(epoch)

        epoch = epoch + 1
    end
end

main()
