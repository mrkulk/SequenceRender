-- Usage: th main.lua
require 'nn'
require 'Bias'
require 'optim'
require 'image'
require 'dataset-mnist'
require 'cutorch'
require 'xlua'
require 'Base'
require 'optim'
require 'image'
trainLogger = optim.Logger(paths.concat('logs/', 'train.log'))
testLogger = optim.Logger(paths.concat('logs/', 'test.log'))

params = {
	bsize = 120,
	image_width = 32,
	template_width = 10,
	num_acrs = 5,
	rnn_size = 100,
	seq_length=1,
	layers=2,
	decay=2,
	dropout=0,
	init_weight=0.1,
	lr=1e-3,
	max_epochs=40,
	max_grad_norm=5
}

config = {
    learningRate = params.lr,
    momentumDecay = 0.1,
    updateDecay = 0.01
}

require 'model'


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
	ret = model:forward({torch.rand(params.bsize, 1, 32, 32):cuda(),
	    {torch.zeros(params.bsize, params.rnn_size):cuda(), torch.zeros(params.bsize, params.rnn_size):cuda(), torch.zeros(params.bsize, params.rnn_size):cuda(), torch.zeros(params.bsize, params.rnn_size):cuda() },
		})
	print(ret[3])

	ret = model:backward({torch.rand(params.bsize, 1, 32, 32):cuda(),
	    {torch.zeros(params.bsize, params.rnn_size):cuda(), torch.zeros(params.bsize, params.rnn_size):cuda(), torch.zeros(params.bsize, params.rnn_size):cuda(), torch.zeros(params.bsize, params.rnn_size):cuda() },
		},
		{
		torch.zeros(1):cuda(),
		{torch.zeros(params.bsize, params.rnn_size):cuda(), torch.zeros(params.bsize, params.rnn_size):cuda(), torch.zeros(params.bsize, params.rnn_size):cuda(), torch.zeros(params.bsize, params.rnn_size):cuda() }
		,torch.rand(params.bsize, 1, 32, 32):cuda()
		})
	-- image.save('test.png',ret[2])
end
-- unit_test()


parts = setup()
function main()
  print("Network parameters:")
  print(params)
  reset_state(state)
  local epoch = 0
  local beginning_time = torch.tic()
  local start_time = torch.tic()
  print("Starting training.")
  print(trainData:size())
  
  for epc = 1,params.max_epochs do
  	print('EPOCH:', epc)
  	local cntr = 0
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
			local perp, output = fp(inputs)
			bp(inputs)
			cutorch.synchronize()
			collectgarbage()	

			if math.fmod(cntr, 1) == 0 then
				-- testing
				for tt = 1,1 do--trainData:size(),params.bsize do
					local inputs = torch.Tensor(params.bsize,1,32,32)
			    local k = 1
			    for ii = tt,math.min(tt+params.bsize-1,testData:size()) do
			       -- load new sample
			       local sample = testData[ii]
			       local input = sample[1]:clone()
			       local _,target = sample[2]:clone():max(1)
			       inputs[{k,1,{},{}}] = input
			       k = k + 1
			    end
			    inputs = inputs:cuda()
					local test_perp, test_output = fp(inputs)

					-- image.save('pred.png', test_output[1][1])
					-- image.save('target.png', inputs[1][1])
		      window1=image.display({image=test_output, nrow=6, legend='Predictions, step : '.. cntr, win=window1})
		      window2=image.display({image=inputs, nrow=6, legend='Targets, step : '.. cntr, win=window2})

		      -- print(test_output)
		      -- exit()

					testLogger:add{['% perp (test set)'] =  test_perp}
					testLogger:style{['% perp (test set)'] = '-'}
					testLogger:plot()
				end
			end

			cntr = cntr + 1
			-- trainLogger:add{['% perp (train set)'] =  perp}
			-- trainLogger:style{['% perp (train set)'] = '-'}
			-- trainLogger:plot()
		end
	end
end

main()
-- print(parts[1].data.module.)