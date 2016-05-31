require 'nn'
require 'cunn'
require 'cutorch'
require 'cudnn'
require 'nngraph'
require 'nnx'
require 'optim'

local createModel = require 'net_graph'
local Trainer = require 'finetrain'
torch.manualSeed(42)

continue = 1
if continue then
--	new = {createModel(), 1, {0, 0, 0, 0.125, 0.25, 0.5}, {}, {}, {}}
	new = torch.load('out/model_223')
end

local model = continue and new[1] or createModel()
local crit = nn.MSECriterion():cuda()
--local loss_w = continue and new[3] or {0, 0, 0, 0.125, 0.25, 0.5}
local loss_w = {1, 0, 0, 0, 0, 0}
local criterion = nn.ParallelCriterion():add(crit, loss_w[1]):add(crit, loss_w[2]):
				add(crit, loss_w[3]):add(crit, loss_w[4]):add(crit, loss_w[5]):
				add(crit, loss_w[6]):cuda()
local optimState = {
		learningRate = 4e-4,
		bata1 = 0.9,
		beta2 = 0.999,
	}


print('loading data')
local x0_te = torch.load('../data.kitti/comp/x0_te')
local x0_tr = torch.load('../data.kitti/comp/x0_tr')
local x1_te = torch.load('../data.kitti/comp/x1_te')
local x1_tr = torch.load('../data.kitti/comp/x1_tr')
local dispnoc_te = torch.load('../data.kitti/comp/dispnoc_te')
local dispnoc_tr = torch.load('../data.kitti/comp/dispnoc_tr')
local metadata_te = torch.load('../data.kitti/comp/metadata_te')
local metadata_tr = torch.load('../data.kitti/comp/metadata_tr')
print('loading data OK')
local trainload = {}
table.insert(trainload, x0_tr)
table.insert(trainload, x1_tr)
table.insert(trainload, dispnoc_tr)
local testload = {}
table.insert(testload, x0_te)
table.insert(testload, x1_te)
table.insert(testload, dispnoc_te)


function savemodel(model, epoch, loss_w, trloss, teloss, prloss)
	local latest = {}
	table.insert(latest, model)
	table.insert(latest, epoch)
	table.insert(latest,loss_w)
	table.insert(latest,trloss)
	table.insert(latest,teloss)
	table.insert(latest,prloss)

	local modelFile = 'out/model_'..epoch
	torch.save(modelFile, latest)
end


local trainer = Trainer(model, criterion, optimState)
local nEpochs = 400
local trloss = continue and new[4] or {}
local teloss = continue and new[5] or {}
local prloss = continue and new[6] or {}
local bestloss = math.huge
local startEpoch = continue and new[2] + 1 or 1

for epoch = startEpoch, nEpochs do
   local trainLoss = trainer:train(epoch, trainload)
   local testLoss = trainer:test(epoch, testload)
   local predLoss = testLoss
   collectgarbage()

   table.insert(trloss, trainLoss)
   table.insert(teloss, testLoss)
   table.insert(prloss, predLoss)
   torch.save('out/trloss', trloss)
   torch.save('out/teloss', teloss)
   torch.save('out/prloss', prloss)

   if testLoss < bestloss then
      bestloss = testLoss
      print(' * Best model ', testLoss)
      savemodel(model, epoch, loss_w, trloss, teloss, prloss)
   end
--   savemodel(model, epoch, loss_w, trloss, teloss, prloss)
end




