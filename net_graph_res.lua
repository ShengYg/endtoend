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
local BN = nn.SpatialBatchNormalization
local ADD = nn.CAddTable
local Ave = nn.SpatialAveragePooling

local M = {}

local function createModel()
	cudnn.fastest = true
	cudnn.benchmark = true

	conv1a = Conv(2, 64, 3, 3, 2, 2, 1, 1)()
	conv1 = ReLU(true)(BN(64)(conv1a))
	conv2a = ReLU(true)(BN(128)(Conv(64, 128, 3, 3, 2, 2, 1, 1)(conv1)))
	conv2b = ReLU(true)(BN(128)(Conv(128, 128, 3, 3, 1, 1, 1, 1)(conv2a)))
	conv2 = ReLU(true)(ADD(true)({conv2b, Ave(1, 1, 2, 2)(conv1)}))
	conv3a = ReLU(true)(BN(256)(Conv(128, 256, 3, 3, 2, 2, 1, 1)(conv2)))
	conv3b = ReLU(true)(BN(256)(Conv(256, 256, 3, 3, 1, 1, 1, 1)(conv3a)))
	conv3 = ReLU(true)(ADD(true)({conv3b, Ave(1, 1, 2, 2)(conv2)}))
	conv4a = ReLU(true)(BN(512)(Conv(256, 512, 3, 3, 2, 2, 1, 1)(conv3)))
	conv4b = ReLU(true)(BN(512)(Conv(512, 512, 3, 3, 1, 1, 1, 1)(conv4a)))
	conv4 = ReLU(true)(ADD(true)({conv4b, Ave(1, 1, 2, 2)(conv3)}))
	conv5a = ReLU(true)(BN(512)(Conv(256, 512, 3, 3, 2, 2, 1, 1)(conv4)))
	conv5b = ReLU(true)(BN(512)(Conv(512, 512, 3, 3, 1, 1, 1, 1)(conv5a)))
	conv5 = ReLU(true)(ADD(true)({conv5b, Ave(1, 1, 2, 2)(conv4)}))
	conv6a = ReLU(true)(BN(1024)(Conv(512, 	1024, 3, 3, 2, 2, 1, 1)(conv5)))
	conv6b = ReLU(true)(BN(1024)(Conv(1024, 1024, 3, 3, 1, 1, 1, 1)(conv6a)))
	conv6 = ReLU(true)(ADD(true)({conv6b, Ave(1, 1, 2, 2)(conv5)}))

	upconv5 = ReLU(true)(BN(512)(DeConv(1024, 512, 4, 4, 2, 2, 1, 1)(conv6)))
	pr6 = Conv(1024, 1, 3, 3, 1, 1, 1, 1)(conv6)
	pr6_1 = ReLU(true)(BN(1)(DeConv(1, 1, 4, 4, 2, 2, 1, 1)(pr6)))
	iconv5 = ReLU(true)(BN(512)(Conv(1025, 512, 3, 3, 1, 1, 1, 1)(ReLU(true)((Join(2)({conv5, pr6_1, upconv5}))))))

	upconv4 = ReLU(true)(BN(256)(DeConv(512, 256, 4, 4, 2, 2, 1, 1)(iconv5)))
	pr5 = Conv(512, 1, 3, 3, 1, 1, 1, 1)(iconv5)
	pr5_1 = ReLU(true)(BN(1)(DeConv(1, 1, 4, 4, 2, 2, 1, 1)(pr5)))
	iconv4 = ReLU(true)(BN(256)(Conv(769, 256, 3, 3, 1, 1, 1, 1)(ReLU(true)((Join(2)({conv4, pr5_1, upconv4}))))))

	upconv3 = ReLU(true)(BN(128)(DeConv(256, 128, 4, 4, 2, 2, 1, 1)(iconv4)))
	pr4 = Conv(256, 1, 3, 3, 1, 1, 1, 1)(iconv4)
	pr4_1 = ReLU(true)(BN(1)(DeConv(1, 1, 4, 4, 2, 2, 1, 1)(pr4)))
	iconv3 = ReLU(true)(BN(128)(Conv(385, 128, 3, 3, 1, 1, 1, 1)(ReLU(true)((Join(2)({conv3, pr4_1, upconv3}))))))

	upconv2 = ReLU(true)(BN(64)(DeConv(128, 64, 4, 4, 2, 2, 1, 1)(iconv3)))
	pr3 = Conv(128, 1, 3, 3, 1, 1, 1, 1)(iconv3)
	pr3_1 = ReLU(true)(BN(1)(DeConv(1, 1, 4, 4, 2, 2, 1, 1)(pr3)))
	iconv2 = ReLU(true)(BN(64)(Conv(193, 64, 3, 3, 1, 1, 1, 1)(ReLU(true)((Join(2)({conv2, pr3_1, upconv2}))))))

	upconv1 = ReLU(true)(BN(32)(DeConv(64, 32, 4, 4, 2, 2, 1, 1)(iconv2)))
	pr2 = Conv(64, 1, 3, 3, 1, 1, 1, 1)(iconv2)
	pr2_1 = ReLU(true)(BN(1)(DeConv(1, 1, 4, 4, 2, 2, 1, 1)(pr2)))
	iconv1 = ReLU(true)(BN(32)(Conv(97, 32, 3, 3, 1, 1, 1, 1)(ReLU(true)((Join(2)({conv1, pr2_1, upconv1}))))))
	pr1 = Conv(32, 1, 3, 3, 1, 1, 1, 1)(iconv1)

	net = nn.gModule({conv1a}, {pr1, pr2, pr3, pr4, pr5, pr6})
	net:cuda()

	return net
end

return createModel