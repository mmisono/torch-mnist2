
-- follow https://github.com/facebook/fb.resnet.torch/blob/master/datasets/README.md

-- https://github.com/deepmind/torch-hdf5
require 'hdf5'

local M = {}
local MnistDataset = torch.class('MnistDataset', M)

function MnistDataset:__init(imageInfo, opt, split)
    --print("reading mnist file... : " .. opt.data) 
    --local f = hdf5.open(opt.data, 'r')
    ---- target is not one-hot
    --self.imageInfo = {
    --    input  = f:read(split .. '/target'):all(),
    --    target = f:read(split .. '/input'):all():add(1)
    --}
    --f:close()
    --print("mnist data load done")
    self.imageInfo = imageInfo[split]
end

function MnistDataset:get(i)
    local input = self.imageInfo.input[i]:float()
    local target = self.imageInfo.target[i]
    
    return {
       input = input,
       target = target,
    }
end

function MnistDataset:size()
    return self.imageInfo.input:size(1)
end

function MnistDataset:preprocess()
    return function(input)
        -- nothing
        -- (assume that input is already normalized)
        return input
    end
end

return M.MnistDataset
