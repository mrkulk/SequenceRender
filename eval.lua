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

params = torch.load('logs/params.t7')

config = {
    learningRate = params.lr,
    momentumDecay = 0.1,
    updateDecay = 0.01
}

require 'model'

-- create test set and normalize
testData = mnist.loadTestSet(nbTestingPatches, geometry)
testData:normalizeGlobal(mean, std)
testLogger = optim.Logger(paths.concat(params.save .. '/', 'test.log'))

setup(true)

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
    window1=image.display({image=test_output, nrow=6, legend='Predictions', win=window1})
    window2=image.display({image=inputs, nrow=6, legend='Targets', win=window2})
    -- window3 = image.display({image=part_images, nrow=3, legend='Strokes', win=window3})
    testLogger:add{['% perp (test set)'] =  test_perp}
    testLogger:style{['% perp (test set)'] = '-'}
    -- testLogger:plot()
  end
end

init()
test()
