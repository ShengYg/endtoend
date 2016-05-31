#! /usr/bin/env luajit

require 'image'
torch.manualSeed(40)

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
width = 768
data_dir = 'data/driving/disparity/bin'
perm = torch.randperm(4400)
--[[
print('partition disp')
disp = torch.FloatTensor(4400, 1, height, width)
disp_tr = torch.FloatTensor(3500, 1, height, width)
disp_te = torch.FloatTensor(900, 1, height, width)
for i = 1, 4400 do
	fname = ('%s/dispnoc_%d.bin'):format(data_dir, i)
	disp[{i}] = fromfile(fname)
	collectgarbage()
end

for i = 1, 3500 do
	local ind = perm[i]
	disp_tr[{i}] = disp[{ind}]
end

for i = 1, 900 do
	local ind = perm[i + 3500]
	disp_te[{i}] = disp[{ind}]
end

torch.save('data/driving/disp_tr', disp_tr)
torch.save('data/driving/disp_te', disp_te)
disp = nil
disp_tr = nil
disp_te = nil
collectgarbage()

print('disptr dowmsample')
disp_tr = torch.load('data/driving/disp_tr')
disp_tr_6 = {}
for i = 1, 3500 do
	--print(i)
	local img = disp_tr[{i}]
	local tab = {}
	--table.insert(tab, img)
	table.insert(tab, image.scale(img, 384, 192))
	table.insert(tab, image.scale(img, 192, 96))
	table.insert(tab, image.scale(img, 96, 48))
	table.insert(tab, image.scale(img, 48, 24))
	table.insert(tab, image.scale(img, 24, 12))
	table.insert(tab, image.scale(img, 12, 6))
	table.insert(disp_tr_6, tab)
	collectgarbage()
end
torch.save('data/driving/disp_tr_6', disp_tr_6)
disp_tr = nil
disp_tr_6 = nil
collectgarbage()


print('dispte dowmsample')
disp_te = torch.load('data/driving/disp_te')
disp_te_6 = {}
for i = 1, 900 do
	--print(i)
	local img = disp_te[{i}]
	local tab = {}
	--table.insert(tab, img)
	table.insert(tab, image.scale(img, 384, 192))
	table.insert(tab, image.scale(img, 192, 96))
	table.insert(tab, image.scale(img, 96, 48))
	table.insert(tab, image.scale(img, 48, 24))
	table.insert(tab, image.scale(img, 24, 12))
	table.insert(tab, image.scale(img, 12, 6))
	table.insert(disp_te_6, tab)
	collectgarbage()
end
torch.save('data/driving/disp_te_6', disp_te_6)
disp_te = nil
disp_te_6 = nil
collectgarbage()
]]

num = {}
for i = 111001, 111300 do
	table.insert(num, i)
end
for i = 112001, 112800 do
	table.insert(num, i)
end
for i = 121001, 121300 do
	table.insert(num, i)
end
for i = 122001, 122800 do
	table.insert(num, i)
end
for i = 211001, 211300 do
	table.insert(num, i)
end
for i = 212001, 212800 do
	table.insert(num, i)
end
for i = 221001, 221300 do
	table.insert(num, i)
end
for i = 222001, 222800 do
	table.insert(num, i)
end

path = 'data/driving'
for _, dir1 in ipairs{'cleanpass_png', 'finalpass_png'} do
	for _, dir2 in ipairs{'left', 'right'} do
		print(('%s, %s'):format(dir1, dir2))
		local x0 = torch.FloatTensor(#num, 1, 384, 768):zero()
		for j, k in ipairs(num) do
			local img_path = '%s/%s/%s/%d.png'
			local img = image.loadPNG(img_path:format(path, dir1, dir2, k), 3, 'byte'):float()
			img = image.rgb2y(img)		--(1, 540, 960)
			img = image.crop(img, 96, 78, 864, 462)
			img:add(-img:mean()):div(img:std())
			x0[{j}]:copy(img)
		end
		torch.save(('%s/%s_%s'):format(path, dir1, dir2), x0)
		collectgarbage()
	end
	collectgarbage()
end


print('partition cleanpass left')
left = torch.load('data/driving/cleanpass_png_left')
left_tr = torch.FloatTensor(3500, 1, height, width)
left_te = torch.FloatTensor(900, 1, height, width)
for i = 1, 3500 do
	local ind = perm[i]
	left_tr[{i}] = left[{ind}]
end
for i = 1, 900 do
	local ind = perm[i + 3500]
	left_te[{i}] = left[{ind}]
end
torch.save('data/driving/clean_left_tr', left_tr)
torch.save('data/driving/clean_left_te', left_te)
left = nil
left_tr = nil
left_te = nil
collectgarbage()


print('partition cleanpass right')
right = torch.load('data/driving/cleanpass_png_right')
right_tr = torch.FloatTensor(3500, 1, height, width)
right_te = torch.FloatTensor(900, 1, height, width)
for i = 1, 3500 do
	local ind = perm[i]
	right_tr[{i}] = right[{ind}]
end
for i = 1, 900 do
	local ind = perm[i + 3500]
	right_te[{i}] = right[{ind}]
end
torch.save('data/driving/clean_right_tr', right_tr)
torch.save('data/driving/clean_right_te', right_te)
right = nil
right_tr = nil
right_te = nil
collectgarbage()

--[[
print('partition finalpass left')
left = torch.load('data/driving/finalpass_png_left')
left_tr = torch.FloatTensor(3500, 1, height, width)
left_te = torch.FloatTensor(900, 1, height, width)
for i = 1, 3500 do
	local ind = perm[i]
	left_tr[{i}] = left[{ind}]
end
for i = 1, 900 do
	local ind = perm[i + 3500]
	left_te[{i}] = left[{ind}]
end
torch.save('data/driving/final_left_tr', left_tr)
torch.save('data/driving/final_left_te', left_te)
left = nil
left_tr = nil
left_te = nil
collectgarbage()


print('partition finalpass right')
right = torch.load('data/driving/finalpass_png_right')
right_tr = torch.FloatTensor(3500, 1, height, width)
right_te = torch.FloatTensor(900, 1, height, width)
for i = 1, 3500 do
	local ind = perm[i]
	right_tr[{i}] = right[{ind}]
end
for i = 1, 900 do
	local ind = perm[i + 3500]
	right_te[{i}] = right[{ind}]
end
torch.save('data/driving/final_right_tr', right_tr)
torch.save('data/driving/final_right_te', right_te)
right = nil
right_tr = nil
right_te = nil
collectgarbage()
]]
--[[
do
	local left = torch.load('data/driving/clean_left_tr')
	local right = torch.load('data/driving/clean_right_tr')
	local disp = torch.load('data/driving/disp_tr_6')
	local trainload = {}
	table.insert(trainload, left)
	table.insert(trainload, right)
	table.insert(trainload, disp)
	torch.save('data/driving/trainload', trainload)
end
collectgarbage()
]]
do
	local left = torch.load('data/driving/clean_left_te')
	local right = torch.load('data/driving/clean_right_te')
	local disp_6 = torch.load('data/driving/disp_te_6')
	local disp = torch.load('data/driving/disp_te')
	local testload = {}
	table.insert(testload, left)
	table.insert(testload, right)
	table.insert(testload, disp_6)
	table.insert(testload, disp)
	torch.save('data/driving/testload', testload)
end
collectgarbage()