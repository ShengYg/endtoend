local optim = require 'optim'
require 'image'

local M = {}
local Trainer = torch.class('net.Trainer', M)
height = 384
width = 768

function databatch(indices,dataloader)
   local sz = indices:size(1)
   local batch = torch.FloatTensor(sz*2, 1, height, width)
   local dispte = {}
   table.insert(dispte, torch.FloatTensor(sz, 1, 192, 384))
   table.insert(dispte, torch.FloatTensor(sz, 1, 96, 192))
   table.insert(dispte, torch.FloatTensor(sz, 1, 48, 96))
   table.insert(dispte, torch.FloatTensor(sz, 1, 24, 48))
   table.insert(dispte, torch.FloatTensor(sz, 1, 12, 24))
   table.insert(dispte, torch.FloatTensor(sz, 1, 6, 12))

   for i, idx in ipairs(indices:totable()) do
      batch[{2 * i - 1}]:copy(dataloader[1][{idx}])
      batch[{2 * i}]:copy(dataloader[2][{idx}])
      for j = 1, 6 do
         dispte[j][{i}]:copy(dataloader[3][idx][j])
      end
      collectgarbage()
   end
   batch = batch:resize(sz, 2, height, width)
   collectgarbage()
   return batch, dispte
end

function databatch_pr(indices,dataloader)
   local sz = indices:size(1)
   local batch = torch.FloatTensor(sz*2, 1, height, width)
   local disppred = torch.FloatTensor(sz, 1, height, width)

   for i, idx in ipairs(indices:totable()) do
      batch[{2 * i - 1}]:copy(dataloader[1][{idx}])
      batch[{2 * i}]:copy(dataloader[2][{idx}])
      disppred[{i}]:copy(dataloader[4][{idx}])
      collectgarbage()
   end
   batch = batch:resize(sz, 2, height, width)
   collectgarbage()
   return batch, disppred
end

function Trainer:__init(model, criterion, optimState)
   self.model = model
   self.criterion = criterion
   self.optimState = optimState or {
      learningRate = 4e-4,
      bata1 = 0.9,
      beta2 = 0.999,
   }
   self.params, self.gradParams = model:getParameters()
end

function Trainer:train(epoch, dataloader)
   -- Trains the model for a single epoch
   self.optimState.learningRate = self:learningRate(epoch)

   local timer = torch.Timer()

   local function feval()
      return self.criterion.output, self.gradParams
   end

   local lossSum, N = 0.0, 0

   print('=> Training epoch # ' .. epoch)
   -- set the batch norm to training mode
   self.model:training()   --!


   local size, batchSize = 3500, 10
   local perm = torch.randperm(size)
   local idx = 1
   
   while idx <= size do
      local indices = perm:narrow(1, idx, math.min(batchSize, size - idx + 1))
      local input, target = databatch(indices, dataloader)

      input = input:cuda()
      for i = 1, 6 do
      	target[i] = target[i]:cuda()
      end

      local output = self.model:forward(input)
      local loss = self.criterion:forward(self.model.output, target)

      self.model:zeroGradParameters()
      self.criterion:backward(self.model.output, target)
      self.model:backward(input, self.criterion.gradInput)

      optim.adam(feval, self.params, self.optimState)

      lossSum = lossSum + math.sqrt(loss)
      N = N + 1

      --print((' | Epoch: [%d]   Time %.3f  Data %.3f  Err %1.4f'):format(epoch,  timer:time().real, dataTime, loss))

      -- check that the storage didn't get changed do to an unfortunate getParameters call
      assert(self.params:storage() == self.model:parameters()[1]:storage())

      idx = idx + batchSize
   end
   print((' | Epoch:[%07d]   LR %.6f   Time %.3f  Err %1.4f'):
   		format(epoch, self.optimState.learningRate, timer:time().real, lossSum/N))

   return lossSum / N
end

function Trainer:test(epoch, dataloader)
   -- Computes the top-1 and top-5 err on the validation set

   local timer = torch.Timer()

   local size, batchSize = 900, 10
   local perm = torch.randperm(size)
   local idx = 1

   local lossSum, N = 0.0, 0
   self.model:evaluate()
   while idx <= size do
      local indices = perm:narrow(1, idx, math.min(batchSize, size - idx + 1))
      local input, target = databatch(indices, dataloader)

      input = input:cuda()
      for i = 1, 6 do
      	target[i] = target[i]:cuda()
      end

      local output = self.model:forward(input)
      local loss = self.criterion:forward(self.model.output, target)

      lossSum = lossSum + math.sqrt(loss)
      N = N + 1

      idx = idx + batchSize
   end
   self.model:training()

	print((' | TEST: [%d]   Time %.3f   Err %1.4f')
			:format(epoch, timer:time().real, lossSum/N))

   timer:reset()

   return lossSum / N
end

function Trainer:pred(epoch, dataloader)
   -- Computes the top-1 and top-5 err on the validation set

   local timer = torch.Timer()

   local size, batchSize = 900, 10
   local perm = torch.randperm(size)
   local idx = 1

   local lossSum, N = 0.0, 0
   self.model:evaluate()
   while idx <= size do
      local indices = perm:narrow(1, idx, math.min(batchSize, size - idx + 1))
      local input, target = databatch_pr(indices, dataloader)

      input = input:cuda()

      local output = self.model:forward(input)[1]:float()
      local output_up = torch.FloatTensor(indices:size(1), 1, height, width)
      for i = 1, indices:size(1) do
      	output_up[i] = image.scale(output[i], width, height)
      end

      local crit = nn.MSECriterion()
      loss = crit:forward(output_up, target)
      lossSum = lossSum + math.sqrt(loss)
      N = N + 1

      idx = idx + batchSize
   end
   self.model:training()

	print((' | PRED: [%d]   Time %.3f   Err %1.4f')
			:format(epoch, timer:time().real, lossSum/N))

   timer:reset()

   return lossSum / N
end

function Trainer:learningRate(epoch)
   -- Training schedule
   return 1e-4/8
end

return M.Trainer
