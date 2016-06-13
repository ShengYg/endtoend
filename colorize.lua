require 'nn'
require 'cunn'
require 'cutorch'
require 'cudnn'
require 'nngraph'
require 'nnx'
require 'optim'
require 'image'
require 'libadcensus'


print('load model')
local new = torch.load('out/model_840')
local model = new[1]
print('load model OK')



local img1 = image.loadPNG('l111001.png', 3, 'byte'):float()
local img2 = image.loadPNG('r111001.png', 3, 'byte'):float()
img1 = image.rgb2y(img1)
img2 = image.rgb2y(img2)
img1 = image.crop(img1, 96, 78, 864, 462)
img2 = image.crop(img2, 96, 78, 864, 462)
img1:add(-img1:mean()):div(img1:std())
img2:add(-img2:mean()):div(img2:std())
local x = torch.FloatTensor(2, 1, 384, 768):zero()
x[{1}]:copy(img1)
x[{2}]:copy(img2)
x = x:resize(1, 2, 384, 768):cuda()
local y = model:forward(x)

local y_col = {}
table.insert(y_col, torch.DoubleTensor(3, 192, 384))
table.insert(y_col, torch.DoubleTensor(3, 96, 192))
table.insert(y_col, torch.DoubleTensor(3, 48, 96))
table.insert(y_col, torch.DoubleTensor(3, 24, 48))
table.insert(y_col, torch.DoubleTensor(3, 12, 24))
table.insert(y_col, torch.DoubleTensor(3, 6, 12))
for i = 1, 6 do
	y[i] = y[i]:double():div(200)
	image.savePNG(('pic/y%d.png'):format(i), y[i][1])
	adcensus.grey2jet(y[i][{1, 1}], y_col[i])
	image.savePNG(('pic/ycol%d.png'):format(i), y_col[i])
end

--[[
local input = {}
table.insert(input, model.modules[])

for i = 1, 6 do
	adcensus.grey2jet(input[i], y_col[i])
	image.savePNG(('pic/downx%d.png'):format(i), y_col[i])
end
]]

