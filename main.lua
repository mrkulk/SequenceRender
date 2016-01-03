-- Tejas D Kulkarni
-- Usage: th main.lua
require 'nn'
require 'randomkit'
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
   -r,--lr            (default 0.0005)       learning rate
   -i,--max_epochs    (default 200)           maximum nb of iterations per batch, for LBFGS
   --bsize            (default 100)           bsize
   --image_width      (default 32)           
   --template_width   (default 10)           
   --num_entities     (default 10)           number of entities
   --rnn_size         (default 100)
   --seq_length       (default 1)
   --layers           (default 1)
   --init_weight      (default 0.1)
   --max_grad_norm    (default 5)
   --dataset          (default "omniglot")
]]
if params.dataset == "omniglot" then
  Entity_FACTOR = 5e3
else
  Entity_FACTOR = 5e3
end
require 'Entity'
config = {
    learningRate = params.lr,
    momentumDecay = 0.1,
    updateDecay = 0.01
}
require 'model'
-- torch.manualSeed(1)

 function normalizeGlobal(data, mean_, std_)
    local std = std_ or data:std()
    local mean = mean_ or data:mean()
    data:add(-mean)
    data:mul(1/std)
    return data
 end



trainLogger = optim.Logger(paths.concat(params.save .. '/', 'train.log'))
testLogger = optim.Logger(paths.concat(params.save .. '/', 'test.log'))

if params.dataset == "omniglot" then
  trainData = torch.load('dataset/omniglot_train_imgs.t7')
  testData = torch.load('dataset/omniglot_test_imgs.t7')

  -- testData = normalizeGlobal(testData, mean, std)
  -- trainData = normalizeGlobal(trainData, mean, std)

  -- fulldata = torch.zeros(trainData:size(1) + testData:size(1), 1, 32,32)
  fulldata = trainData:clone()
  -- fulldata[{{trainData:size(1)+1,testData:size(1)+trainData:size(1) },{},{},{}}] = testData:clone()
else
  --single mnist
  -- create training set and normalize
  trainData = mnist.loadTrainSet(nbTrainingPatches, geometry)
  trainData.data = trainData.data/255
  -- trainData:normalizeGlobal(mean, std)
  -- create test set and normalize
  testData = mnist.loadTestSet(nbTestingPatches, geometry)
  -- testData.data = testData.data/255
  -- testData:normalizeGlobal(mean, std)
  fulldata = torch.zeros(trainData.data:size(1) + testData.data:size(1), 1, 32,32)
  fulldata[{{1,trainData.data:size(1)},{},{},{}}] = trainData.data:clone()
  fulldata[{{trainData.data:size(1)+1,testData.data:size(1)+trainData.data:size(1) },{},{},{}}] = testData.data:clone()
  -- trainData.data[torch.le(trainData.data,0.5)] = 0
  -- trainData.data[torch.ge(trainData.data,0.5)] = 1
  -- testData.data[torch.le(testData.data,0.5)] = 0
  -- testData.data[torch.ge(testData.data,0.5)] = 1
end

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
  for i = t,math.min(t+params.bsize-1,data:size(1)) do
     -- load new sample
     local sample = data[i]
     local input = sample[1]:clone()
     -- local _,target = sample[2]:clone():max(1)
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
  print(fulldata:size())
end

function train()
  for epc = 1,params.max_epochs do
    print('epoch #', epc)
    local cntr = 0
    torch.save(params.save .. '/network.t7', model.rnns[1])
    torch.save(params.save .. '/params.t7', params)
    for t = 1,fulldata:size(1),params.bsize do
      xlua.progress(t, fulldata:size(1))
      -- create mini batch
      local inputs = get_batch(t, fulldata)
      local perp, output = fp(inputs)
      bp(inputs)
      cutorch.synchronize()
      collectgarbage()  

      if params.plot and math.fmod(cntr, 20) == 0  then 
        test()
      end

      cntr = cntr + 1
      trainLogger:add{['% perp (train set)'] =  perp}
      trainLogger:style{['% perp (train set)'] = '-'}
      -- trainLogger:plot()
    end
    -- params.lr = params.lr * 0.9
    -- Entity_FACTOR = Entity_FACTOR * 0.8
  end
end

function test()
  -- testing
  -- g_disable_dropout(model.rnns)
  local test_err = 0
  for tt = 1,1 do--trainData:size(),params.bsize do
    local inputs = get_batch(tt, testData)
    local test_perp, test_output = fp(inputs)
    test_err = test_perp + test_err
    local entity_imgs = {}; entity_fg_imgs={};
    for pp = 1,params.num_entities do
      entity_imgs[pp] = extract_node(model.rnns[1], 'entity_' .. pp).data.module.output:double()
      -- entity_fg_imgs[pp] = extract_node(model.rnns[1], 'entity_fg_' .. pp).data.module.output:double()
    end
    local en_imgs = {}; en_fg_imgs={};
    counter=1
    for bb = 1,MAX_IMAGES_TO_DISPLAY do
      for pp=1,params.num_entities do
        en_imgs[counter] =entity_imgs[pp][bb]
        -- en_fg_imgs[counter] = entity_fg_imgs[pp][bb]
        counter = counter + 1 
      end
    end
    if params.plot then
      window1=image.display({image=test_output[{{1,MAX_IMAGES_TO_DISPLAY},{},{},{}}], nrow=1, legend='Predictions', win=window1})
      window2=image.display({image=inputs[{{1,MAX_IMAGES_TO_DISPLAY},{},{},{}}], nrow=1, legend='Targets', win=window2})

      window3=image.display({image=en_imgs, nrow=params.num_entities, legend='Entities', win=window3})
      -- window4=image.display({image=en_fg_imgs, nrow=params.num_entities, legend='FG', win=window4})
      -- window3 = image.display({image=part_images, nrow=3, legend='Strokes', win=window3})
    end
  end
  testLogger:add{['% perp (test set)'] =  test_err}
  testLogger:style{['% perp (test set)'] = '-'}
  -- testLogger:plot()
  -- g_enable_dropout(model.rnns)
end

MAX_IMAGES_TO_DISPLAY = 20
init()
train()
