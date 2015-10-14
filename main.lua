-- Usage: th main.lua


package.path = package.path .. ';modules/Affine/?.lua'
require 'nn'
require 'INTM'
require 'ACR'
require 'Bias'
require 'optim'
require 'image'
require 'dataset-mnist'
require 'cutorch'
require 'xlua'
require 'Base'
require 'optim'

trainLogger = optim.Logger(paths.concat('logs/', 'train.log'))
testLogger = optim.Logger(paths.concat('logs/', 'test.log'))

params = {
	bsize = 3,
	image_width = 32,
	template_width = 10,
	num_acrs = 3,
	rnn_size = 200,
	seq_length=1,
	layers=2,
	decay=2,
	dropout=0,
	init_weight=0.1,
	lr=1e1,
	max_epochs=4,
	max_grad_norm=5
}

require 'model'

setup()

-- torch.manualSeed(1)

-- create training set and normalize
trainData = mnist.loadTrainSet(nbTrainingPatches, geometry)
trainData:normalizeGlobal(mean, std)

-- create test set and normalize
testData = mnist.loadTestSet(nbTestingPatches, geometry)
testData:normalizeGlobal(mean, std)

local function unit_test()
	model = create_network(params)
	model:cuda()
	ret = model:forward({torch.rand(params.bsize, 1, 32, 32):cuda(),torch.rand(params.bsize, 1, 32, 32):cuda(),
	    {torch.zeros(params.bsize, params.rnn_size):cuda(), torch.zeros(params.bsize, params.rnn_size):cuda(), torch.zeros(params.bsize, params.rnn_size):cuda(), torch.zeros(params.bsize, params.rnn_size):cuda() },
		})
	-- print(ret)

	ret = model:backward({torch.rand(params.bsize, 1, 32, 32):cuda(),torch.rand(params.bsize, 1, 32, 32):cuda(),
	    {torch.zeros(params.bsize, params.rnn_size):cuda(), torch.zeros(params.bsize, params.rnn_size):cuda(), torch.zeros(params.bsize, params.rnn_size):cuda(), torch.zeros(params.bsize, params.rnn_size):cuda() },
		},
		{
		torch.zeros(1):cuda(),
		{torch.zeros(params.bsize, params.rnn_size):cuda(), torch.zeros(params.bsize, params.rnn_size):cuda(), torch.zeros(params.bsize, params.rnn_size):cuda(), torch.zeros(params.bsize, params.rnn_size):cuda() }
		,torch.rand(params.bsize, 1, 32, 32):cuda()
		})

	-- image.save('test.png',ret[2])
end
unit_test()



function main()
  print("Network parameters:")
  print(params)
  reset_state(state)
  local epoch = 0
  local beginning_time = torch.tic()
  local start_time = torch.tic()
  print("Starting training.")
  print(trainData:size())
	for t = 1,trainData:size(),params.bsize do
			xlua.progress(t, trainData:size())
      -- create mini batch
      local inputs = torch.Tensor(params.bsize,1,32,32)
      local k = 1
      for i = t,math.min(t+params.bsize-1,trainData:size()) do
         -- load new sample
         local sample = trainData[i]
         local input = sample[1]:clone()
         local _,target = sample[2]:clone():max(1)
         inputs[{k,1,{},{}}] = input
         k = k + 1
      end
    inputs = inputs:cuda()
		local perp = fp(inputs)
		bp(inputs)
		cutorch.synchronize()
		collectgarbage()

		trainLogger:add{['% perp (train set)'] =  perp}
		trainLogger:style{['% perp (train set)'] = '-'}
		trainLogger:plot()
	end
end

-- main()