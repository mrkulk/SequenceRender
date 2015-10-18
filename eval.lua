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
matio = require 'matio'

plot = false
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
  local targets = torch.Tensor(params.bsize)
  local k = 1
  for i = t,math.min(t+params.bsize-1,data:size()) do
     -- load new sample
     local sample = data[i]
     local input = sample[1]:clone()
     local _,target = sample[2]:clone():max(1)
     inputs[{k,1,{},{}}] = input
     targets[k] = target
     k = k + 1
  end
  inputs = inputs:cuda()
  return inputs, targets
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
  -- for tt = 1,1 do--trainData:size(),params.bsize do
  bid = 1
  for tt = 1, max_num,params.bsize do
    local inputs, targets = get_batch(tt, testData)
    local test_perp, test_output = fp(inputs)
    local affines = {}
    local entity_imgs = {}
    -- local part_images = {}
    for pp = 1,params.num_entities do
    --   local p1_images = entities[pp].data.module.bias[1]:reshape(params.template_width, params.template_width)
    --   part_images[pp] = p1_images
      affines[pp] = extract_node(model.rnns[1], 'affines_' .. pp).data.module.output:double() 
      entity_imgs[pp] = extract_node(model.rnns[1], 'entity_' .. pp).data.module.output:double()
      matio.save('tmp/batch_aff_'.. pp ..'_' .. bid, {aff = affines[pp]})
      matio.save('tmp/batch_ent_'.. pp ..'_' .. bid, {entity = entity_imgs[pp]})
    end
      
    local en_imgs = {}
    counter=1
    for bb = 1,MAX_IMAGES_TO_DISPLAY do
      for pp=1,params.num_entities do
        en_imgs[counter] =entity_imgs[pp][bb]
        counter = counter + 1 
      end
    end

    if plot then
      window1=image.display({image=test_output[{{1,MAX_IMAGES_TO_DISPLAY},{},{},{}}], nrow=1, legend='Predictions', win=window1})
      window2=image.display({image=inputs[{{1,MAX_IMAGES_TO_DISPLAY},{},{},{}}], nrow=1, legend='Targets', win=window2})
      window3=image.display({image=en_imgs, nrow=params.num_entities, legend='Entities', win=window3})
      -- window3 = image.display({image=part_images, nrow=3, legend='Strokes', win=window3})
    end
    testLogger:add{['% perp (test set)'] =  test_perp}
    testLogger:style{['% perp (test set)'] = '-'}
    -- testLogger:plot()

    matio.save('tmp/batch_imgs_' .. bid, {imgs = inputs:double()})
    matio.save('tmp/batch_labels_' .. bid, {labels = targets:double()})
    
    bid = bid + 1
  end
end

MAX_IMAGES_TO_DISPLAY = 10
plot = false
if plot then
  max_num = 1
else
  max_num = testData:size()
end
init()
test()
-- print(extract_node(model.rnns[1], 'entity_1').data.module.output[1][1])