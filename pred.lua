require 'nn'
require 'cunn'
require 'cutorch'
require 'cudnn'
require 'nngraph'
require 'nnx'
require 'optim'
require 'image'


function fromfile(fname)
	local file = io.open(fname .. '.dim')
	local dim = {}
	for line in file:lines() do
		table.insert(dim, tonumber(line))
	end
	if #dim == 1 and dim[1] == 0 then
		return torch.Tensor()
	end
	local x = torch.FloatTensor(torch.FloatStorage(fname)) 
	x = x:reshape(torch.LongStorage(dim))
	return x
end

height = 384
width = 1280

print('load model')
local new = torch.load('out/model_223')
local model = new[1]
print('load model OK')
--local path1 = 'data/driving/cleanpass_png/left/'
--local path2 = 'data/driving/cleanpass_png/right/'
--local img1 = image.loadPNG(path1 .. '111001.png', 3, 'byte'):float()
--local img2 = image.loadPNG(path2 .. '111001.png', 3, 'byte'):float()

--todo == 1 ==> predict an image
--tod0 == 2 ==> test the err described in kitti dataset
local todo = 2
if todo == 1 then
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
	local y = model:forward(x)[1]:float()
	local y_up = image.scale(y[{1}], 768, 384)
	print ('dispmax==>'..y_up:max())
	local N = 200
	image.savePNG('p111001.png', y_up:div(N))

	--[[
	data_dir = 'data/driving/disparity/bin'
	fname = ('%s/dispnoc_1.bin'):format(data_dir)
	disp = fromfile(fname)
	]]
elseif todo == 2 then
	local path1 = '../data.kitti/comp/'
	local path2 = '../data.kitti2015/comp/'
	local X0 = torch.load(path1 .. 'x0')
	local X1 = torch.load(path1 .. 'x1')
	local dispnoc = torch.load(path1 .. 'dispnoc')
	local te = torch.load(path1 .. 'te')
	local metadata = torch.load(path1 .. 'metadata')
	local examples = {}
	for i = 1,40 do
		table.insert(examples, te[i])
	end

	local pred_bad = torch.FloatTensor()
	local err_sum = 0
	for i, k in ipairs(examples) do
		local input = torch.FloatTensor(2, 1, height, width)
		print(i)
		local img_height = metadata[{i,1}]
		local img_width = metadata[{i,2}]
		input[1] = X0[i]
		input[2] = X1[i]
		input = input:resize(1, 2, height, width):cuda()

		local output = model:forward(input)[1]:float()
		local pred = image.scale(output[{1}], width, height)[{{},{1,img_height},{1,img_width}}]

		local actual = dispnoc[i][{{},{1,img_height},{1,img_width}}]
		pred_bad:resizeAs(actual)
		local mask = torch.FloatTensor():resizeAs(actual):ne(actual, 0)
		actual:add(-1, pred):abs()
		pred_bad:gt(actual, 3):cmul(mask)
		local err = pred_bad:sum() / mask:sum()
		err_sum = err_sum + err
		print(err)
	end
	print('err total ',err_sum / #examples)
end


