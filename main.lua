-- Tejas D Kulkarni
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
require 'sys'
require 'pl'

params = lapp[[
   -s,--save          (default "logs")      subdirectory to save logs
   -m,--model         (default "convnet")   type of model tor train: convnet | mlp | linear
   -p,--plot                                plot while training
   -r,--lr            (default 0.005)       learning rate
   -i,--max_epochs    (default 50)           maximum nb of iterations per batch, for LBFGS
   --bsize            (default 120)           bsize
   --image_width      (default 32)           
   --template_width   (default 10)           
   --num_entities           (default 3)           number of acrs
   --rnn_size         (default 100)
   --seq_length       (default 1)
   --layers           (default 1)
   --init_weight      (default 0.1)
   --max_grad_norm    (default 5)
]]
config = {
    learningRate = params.lr,
    momentumDecay = 0.1,
    updateDecay = 0.01
}
require 'model'
-- torch.manualSeed(1)
trainLogger = optim.Logger(paths.concat(params.save .. '/', 'train.log'))
testLogger = optim.Logger(paths.concat(params.save .. '/', 'test.log'))

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
end
-- unit_test()

setup(false)

function get_batch(t, data)
  local inputs = torch.Tensor(params.bsize,1,32,32)
  local k = 1
  for i = t,math.min(t+params.bsize-1,data:size()) do
     -- load new sample
     local sample = data[i]
     local input = sample[1]:clone()
     local _,target = sample[2]:clone():max(1)
     inputs[{k,1,{},{}}] = input
     k = k + 1
  end
  inputs = inputs:cuda()
  return inputs
end

function init()
  print("Network parameters:")
  print(params)
  reset_state(state)
  local epoch = 0
  local beginning_time = torch.tic()
  local start_time = torch.tic()
  print("Starting training.")
  print(trainData:size())
end

function train()
  for epc = 1,params.max_epochs do
    print('epoch #', epc)
    local cntr = 0
    for t = 1,trainData:size(),params.bsize do
      xlua.progress(t, trainData:size())
      -- create mini batch
      local inputs = get_batch(t, trainData)
      local perp, output = fp(inputs)
      bp(inputs)
      cutorch.synchronize()
      collectgarbage()  

      if math.fmod(cntr, 50000) == 0 then
        -- test()
        torch.save(params.save .. '/network.t7', model.rnns[1])
        torch.save(params.save .. '/params.t7', params)
      end

      cntr = cntr + 1
      trainLogger:add{['% perp (train set)'] =  perp}
      trainLogger:style{['% perp (train set)'] = '-'}
    end
  end
end

function test()
  -- testing
  for tt = 1,1 do--trainData:size(),params.bsize do
    local inputs = get_batch(tt, testData)
    local test_perp, test_output = fp(inputs)
    -- local part_images = {}
    -- for pp = 1,params.num_entities do
    --   local p1_images = entities[pp].data.module.bias[1]:reshape(params.template_width, params.template_width)
    --   part_images[pp] = p1_images
    -- end
    if params.plot then 
      window1=image.display({image=test_output, nrow=6, legend='Predictions', win=window1})
      window2=image.display({image=inputs, nrow=6, legend='Targets', win=window2})
      -- window3 = image.display({image=part_images, nrow=3, legend='Strokes', win=window3})
    end
    testLogger:add{['% perp (test set)'] =  test_perp}
    testLogger:style{['% perp (test set)'] = '-'}
    -- testLogger:plot()
  end
end

init()
train()
