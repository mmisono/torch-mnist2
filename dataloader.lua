
-- based on https://github.com/facebook/fb.resnet.torch/blob/master/dataloader.lua
--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--          

local threads = require 'threads' -- torch/threads
threads.serialization('threads.sharedserialize')

local M = {}
local DataLoader = torch.class('DataLoader', M)

function DataLoader.create(opt)
    -- The train and val loader
    local loaders = {}
    local Dataset = require('datasets/' .. opt.dataset)
    local hdf5 = require "hdf5"
    local f = hdf5.open(opt.data, 'r')
    local imageInfo = {
        train = {
            input  = f:read('train/input'):all(),
            target = f:read('train/target'):all():add(1)
            --target = f:read('train/target'):all()
        },
        val = {
            input  = f:read('val/input'):all(),
            target = f:read('val/target'):all():add(1)
        }
    }
    f:close()

    for i, split in ipairs{'train', 'val'} do
        local data = Dataset(imageInfo, opt, split)
        loaders[i] = M.DataLoader(data, opt)
    end

    return table.unpack(loaders)
end

function DataLoader:__init(dataset, opt)
    local function init()
        require('datasets/' .. opt.dataset)
    end
    local function main(idx)
        -- enable to use dataset in each threads
        _G.dataset = dataset
        _G.preprocess = dataset:preprocess()
        -- return dataset:size()
    end

    -- each threads run init & main then return dataset:size()
    -- local ths, sizes = threads.Threads(opt.nThreads, init, main)
    print("create threads")
    local ths = threads.Threads(opt.nThreads, init, main)
    print("done")
    self.threads = ths
    -- self.__size = sizes[1][1]
    self.__size = dataset:size()
    self.batchSize = opt.batchSize
end

function DataLoader:size()
   return math.ceil(self.__size / self.batchSize)
end

function DataLoader:run()
   local threads = self.threads
   local size, batchSize = self.__size, self.batchSize
   local perm = torch.randperm(size)

   local idx, sample = 1, nil
   local function enqueue()
      while idx <= size and threads:acceptsjob() do
         local indices = perm:narrow(1, idx, math.min(batchSize, size - idx + 1))
         threads:addjob(
            function(indices)
               local sz = indices:size(1)
               local batch, imageSize
               local target = torch.IntTensor(sz)
               for i, idx in ipairs(indices:totable()) do
                  local sample = _G.dataset:get(idx)
                  local input = _G.preprocess(sample.input)
                  if not batch then
                     imageSize = input:size():totable()
                     batch = torch.FloatTensor(sz, 1, table.unpack(imageSize))
                  end
                  batch[i]:copy(input)
                  target[i] = sample.target
               end
               collectgarbage()
               return {
                  input = batch:view(sz, table.unpack(imageSize)),
                  target = target,
               }
            end,
            function(_sample_)
               sample = _sample_
            end,
            indices
         )
         idx = idx + batchSize
      end
   end

   local n = 0
   local function loop()
      enqueue()
      if not threads:hasjob() then
         return nil
      end
      threads:dojob()
      if threads:haserror() then
         threads:synchronize()
      end
      enqueue()
      n = n + 1
      return n, sample
   end

   return loop
end

return M.DataLoader
