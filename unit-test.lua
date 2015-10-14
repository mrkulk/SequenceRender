require 'nn'
require 'stn'
require 'cudnn'
require 'image'
require 'nngraph'

function get_transformer()
	local x = nn.Identity()()
	local encoder_out = nn.Identity()()

	local outLayer = nn.Linear(20,6)(encoder_out)
	outLayer.data.module.weight:fill(0)
	local bias = torch.FloatTensor(6):fill(0)
	bias[1]=2
	bias[5]=2
	outLayer.data.module.bias:copy(bias)
	-- there we generate the grids
	local grid = nn.AffineGridGeneratorBHWD(32,32)(nn.View(2,3)(outLayer))

	-- first branch is there to transpose inputs to BHWD, for the bilinear sampler
	local tranet=nn.Transpose({2,3},{3,4})(x)

	local spanet = nn.BilinearSamplerBHWD()({tranet, grid})
	local sp_out = nn.Transpose({3,4},{2,3})(spanet)
	return nn.gModule({x, encoder_out}, {sp_out})
end

transformer = get_transformer()
transformer:cuda()
ret = transformer:forward({torch.rand(256,1,32,32):cuda(), torch.rand(256, 20):cuda()})
image.display({image=ret, nrow=16})




local function my_imp() 
	require 'nn'
	package.path = package.path .. ';modules/Affine/?.lua'

	require 'INTM'
	require 'ACR'
	require 'Bias'
	require 'optim'
	require 'image'


	twidth = 10
	imwidth = 32
	acr = nn.ACR(1,imwidth)
	intm = nn.INTM(1,7,10, imwidth)
	bias = nn.Bias(1, twidth*twidth)


	net = nn.Sequential()
	acr_wrapper = nn.Sequential()
	acr_wrapper:add(nn.Replicate(2))
	acr_wrapper:add(nn.SplitTable(1))

	acr_in = nn.ParallelTable()
	acr_in:add(bias)
	acr_in:add(intm)

	acr_wrapper:add(acr_in)
	net:add(acr_wrapper)
	net:add(acr)

	data = torch.Tensor({0, 5 , 1, 0.89, 0.5, -1.40, 1})
	data = data:reshape(1,7)
	targets = torch.zeros(imwidth,imwidth):reshape(1,imwidth,imwidth)
	outputs = net:forward(data):clone()
	image.save('test.png', outputs[1])
end
