require 'optim'

local M = {}
local Trainer = torch.class('Trainer', M)

function Trainer:__init(model, criterion, opt)
    self.model = model
    self.criterion = criterion
    self.opt = opt
    self.params, self.gradParams = model:getParameters()

    self.optimState = {
        learningRate = opt.learningRate,
        weightDecay = opt.weightDecay,
        learningRateDecay = opt.learningRateDecay,
        momentum = opt.momentum,
        nestrov = true,
    }

    if self.opt.confusion then
        self.confusion = optim.ConfusionMatrix(self.opt.classes)
        self.confusion:zero()
    end
end


function Trainer:train(epoch, dataloader)
    self.model:training()
    trainSize = dataloader:size()
    local timer = torch.Timer()

    function feval()
        return self.criterion.output, self.gradParams
    end

    print('=> Training epoch # ' .. epoch)
    -- dataloader return mini-batch sample
    for n, sample in dataloader:run() do

        self:copyInputs(sample)
    
        local output = self.model:forward(self.input):float()
        local loss = self.criterion:forward(self.model.output, self.target)

        self.model:zeroGradParameters()
        self.criterion:backward(self.model.output, self.target)
        self.model:backward(self.input, self.criterion.gradInput)

        optim.sgd(feval, self.params, self.optimState)
    
        if self.opt.confusion then
            self.confusion:batchAdd(output, self.target)
        end
    end
    if self.opt.confusion then
        print(self.confusion)
        self.confusion:zero()
    end
end

function Trainer:copyInputs(sample)
    --self.input = self.input or (self.opt.nGPU == 1
    --   and torch.CudaTensor()
    --   or cutorch.createCudaHostTensor())
    self.input = self.input or torch.CudaTensor()
    self.target = self.target or torch.CudaTensor()

    self.input:resize(sample.input:size()):copy(sample.input)
    self.target:resize(sample.target:size()):copy(sample.target)
end

function Trainer:test(epoch, dataloader)
    self.model:evaluate()
    testSize = dataloader:size()
    for n, sample in dataloader:run() do
        self:copyInputs(sample)
    
        local output = self.model:forward(self.input):float()
        local loss = self.criterion:forward(self.model.output, self.target)

        if self.opt.confusion then
            self.confusion:batchAdd(output, self.target)
        end
    end

    if self.opt.confusion then
        print(self.confusion)
        self.confusion:zero()
    end
end

return M.Trainer
