require 'nn'
require 'cunn'
require 'cutorch'
require 'cudnn'
require 'nngraph'
require 'nnx'
require 'optim'

local createModel = require 'net_graph'
local Trainer = require 'train'
torch.manualSeed(40)

continue = 0
if continue then
--	new = {createModel(), 1, {0, 0, 0, 0.125, 0.25, 0.5}, {}, {}, {}}
	new = torch.load('out/model_840')
end

local model = continue == 1 and new[1] or createModel()
local crit = nn.MSECriterion():cuda()
--local loss_w = {0.5, 0.5, 0.5, 0.5, 0.5, 0.5}
local loss_w = {0, 0, 0, 0.125, 0.25, 0.5}
local criterion = nn.ParallelCriterion():add(crit, loss_w[1]):add(crit, loss_w[2]):
				add(crit, loss_w[3]):add(crit, loss_w[4]):add(crit, loss_w[5]):
				add(crit, loss_w[6]):cuda()
local optimState = {
		learningRate = 4e-4,
		bata1 = 0.9,
		beta2 = 0.999,
	}


print('loading data')
local trainload = torch.load('data/driving/trainload')
local testload = torch.load('data/driving/testload')
print('loading data OK')

--[[
function clean_net_sub(net)
	net.output = torch.CudaTensor()
	net.gradInput = nil
	net.gradWeight = nil
	net.gradBias = nil
end

function clean_net(net)
	net.output = {torch.CudaTensor(),torch.CudaTensor(),torch.CudaTensor(),
					torch.CudaTensor(),torch.CudaTensor(),torch.CudaTensor()}
	net.gradInput = nil
	net.gradWeight = nil
	net.gradBias = nil
	if net.modules then
		for _, module in ipairs(net.modules) do
			clean_net_sub(module)
		end
	end
	return net
end
]]

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
local nEpochs = 100
local trloss = continue == 1 and new[4] or {}
local teloss = continue == 1 and new[5] or {}
local prloss = continue == 1 and new[6] or {}
local bestloss = math.huge
local startEpoch = continue == 1 and new[2] + 1 or 1

for epoch = startEpoch, nEpochs do
	local trainLoss = trainer:train(epoch, trainload)
	local testLoss = trainer:test(epoch, testload)
	local predLoss = trainer:pred(epoch, testload)
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
--		savemodel(model, epoch, loss_w, trloss, teloss, prloss)
	end
	if epoch%10 == 0 then
   		savemodel(model, epoch, loss_w, trloss, teloss, prloss)
   	end
end




