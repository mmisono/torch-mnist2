require 'optim'

local M = {}
local Trainer = torch.class('Trainer', M)

function Trainer:__init(model, criterion, opt, trainData, valData)
    self.model = model
    self.criterion = criterion
    self.opt = opt
    self.params, self.gradParams = model:getParameters()
    self.trainData = trainData
    self.valData = valData

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


function Trainer:train(epoch)
    self.model:training()
    trainSize = self.trainData:size()
    local shuffle = torch.randperm(trainSize)
    local timer = torch.Timer()

    function feval()
        return self.criterion.output, self.gradParams
    end

    print('=> Training epoch # ' .. epoch)
    local n = 0
    for t = 1,trainSize,self.opt.batchSize do
        if t + self.opt.batchSize-1 > trainSize then
          break
        end
        n = n + self.opt.batchSize
    
        xlua.progress(t, trainSize)
    
        -- create mini batch
        local input = torch.Tensor(self.opt.batchSize, 1, 32, 32) -- input size: 32x32
        local target = torch.Tensor(self.opt.batchSize)
        for i = t, t+self.opt.batchSize-1 do
            local sample = self.trainData[shuffle[i]]
            local _, target_ = sample[2]:clone():max(1)
            target_ = target_:squeeze()
            input[i-t+1] = sample[1]:clone()
            target[i-t+1] = target_
        end
        if self.opt.backend == 'cudnn' or self.opt.backend == 'cunn' then
            input = input:cuda()
            target = target:cuda()
        end

        local output = self.model:forward(input)
        local loss = self.criterion:forward(self.model.output, target)

        self.model:zeroGradParameters()
        self.criterion:backward(self.model.output, target)

        self.model:backward(input, self.criterion.gradInput)
        self.criterion:backward(output, target)
        optim.sgd(feval, self.params, self.optimState)
    
        if self.opt.confusion then
            self.confusion:batchAdd(output, target)
        end
    end
    if self.opt.confusion then
        print(self.confusion)
        self.confusion:zero()
    end
end


function Trainer:test(epoch)
end

return M.Trainer
