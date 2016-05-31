require 'nn'
require 'cutorch'
require 'cudnn'
require 'nngraph'
require 'nnx'

local Conv = cudnn.SpatialConvolution
local ReLU = cudnn.ReLU
local Concat = nn.ConcatTable
local DeConv = cudnn.SpatialFullConvolution
local Join = nn.JoinTable



--[[disp1 = nn.SpatialReSampling{owidth=384,oheight=192}()
disp2 = nn.SpatialReSampling{owidth=192,oheight=96}(disp1)
disp3 = nn.SpatialReSampling{owidth=96,oheight=48}(disp2)
disp4 = nn.SpatialReSampling{owidth=48,oheight=24}(disp3)
disp5 = nn.SpatialReSampling{owidth=24,oheight=12}(disp4)
disp6 = nn.SpatialReSampling{owidth=12,oheight=6}(disp5)]]
local M = {}

local function createModel()
	cudnn.fastest = true
	cudnn.benchmark = true

	conv1 = Conv(2, 64, 3, 3, 2, 2, 1, 1)()
	conv2 = Conv(64, 128, 3, 3, 2, 2, 1, 1)(ReLU(true)(conv1))
	conv3a = Conv(128, 256, 3, 3, 2, 2, 1, 1)(ReLU(true)(conv2))
	conv3b = Conv(256, 256, 3, 3, 1, 1, 1, 1)(ReLU(true)(conv3a))
	conv4a = Conv(256, 512, 3, 3, 2, 2, 1, 1)(ReLU(true)(conv3b))
	conv4b = Conv(512, 512, 3, 3, 1, 1, 1, 1)(ReLU(true)(conv4a))
	conv5a = Conv(512, 512, 3, 3, 2, 2, 1, 1)(ReLU(true)(conv4b))
	conv5b = Conv(512, 512, 3, 3, 1, 1, 1, 1)(ReLU(true)(conv5a))
	conv6a = Conv(512, 1024, 3, 3, 2, 2, 1, 1)(ReLU(true)(conv5b))
	conv6b = Conv(1024, 1024, 3, 3, 1, 1, 1, 1)(ReLU(true)(conv6a))

	upconv5 = DeConv(1024, 512, 4, 4, 2, 2, 1, 1)(ReLU(true)(conv6b))
	pr6 = Conv(1024, 1, 3, 3, 1, 1, 1, 1)(ReLU(true)(conv6b))
	pr6_1 = DeConv(1, 1, 4, 4, 2, 2, 1, 1)(ReLU(true)(pr6))
	iconv5 = Conv(1025, 512, 3, 3, 1, 1, 1, 1)(ReLU(true)((Join(2)({ReLU(true)(conv5b), ReLU(true)(pr6_1), ReLU(true)(upconv5)}))))

	upconv4 = DeConv(512, 256, 4, 4, 2, 2, 1, 1)(ReLU(true)(iconv5))
	pr5 = Conv(512, 1, 3, 3, 1, 1, 1, 1)(ReLU(true)(iconv5))
	pr5_1 = DeConv(1, 1, 4, 4, 2, 2, 1, 1)(ReLU(true)(pr5))
	iconv4 = Conv(769, 256, 3, 3, 1, 1, 1, 1)(ReLU(true)((Join(2)({ReLU(true)(conv4b), ReLU(true)(pr5_1), ReLU(true)(upconv4)}))))

	upconv3 = DeConv(256, 128, 4, 4, 2, 2, 1, 1)(ReLU(true)(iconv4))
	pr4 = Conv(256, 1, 3, 3, 1, 1, 1, 1)(ReLU(true)(iconv4))
	pr4_1 = DeConv(1, 1, 4, 4, 2, 2, 1, 1)(ReLU(true)(pr4))
	iconv3 = Conv(385, 128, 3, 3, 1, 1, 1, 1)(ReLU(true)((Join(2)({ReLU(true)(conv3b), ReLU(true)(pr4_1), ReLU(true)(upconv3)}))))

	upconv2 = DeConv(128, 64, 4, 4, 2, 2, 1, 1)(ReLU(true)(iconv3))
	pr3 = Conv(128, 1, 3, 3, 1, 1, 1, 1)(ReLU(true)(iconv3))
	pr3_1 = DeConv(1, 1, 4, 4, 2, 2, 1, 1)(ReLU(true)(pr3))
	iconv2 = Conv(193, 64, 3, 3, 1, 1, 1, 1)(ReLU(true)((Join(2)({ReLU(true)(conv2), ReLU(true)(pr3_1), ReLU(true)(upconv2)}))))

	upconv1 = DeConv(64, 32, 4, 4, 2, 2, 1, 1)(ReLU(true)(iconv2))
	pr2 = Conv(64, 1, 3, 3, 1, 1, 1, 1)(ReLU(true)(iconv2))
	pr2_1 = DeConv(1, 1, 4, 4, 2, 2, 1, 1)(ReLU(true)(pr2))
	iconv1 = Conv(97, 32, 3, 3, 1, 1, 1, 1)(ReLU(true)((Join(2)({ReLU(true)(conv1), ReLU(true)(pr2_1), ReLU(true)(upconv1)}))))
	pr1 = Conv(32, 1, 3, 3, 1, 1, 1, 1)(ReLU(true)(iconv1))

	net = nn.gModule({conv1}, {pr1, pr2, pr3, pr4, pr5, pr6})
	net:cuda()

	return net
end

return createModel