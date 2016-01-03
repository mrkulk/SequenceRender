-- Tejas D Kulkarni
require 'nn'
require 'Entity'
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
require 'paths'
matio = require 'matio'
src = 'slurm_logs/omni__num_entities_20_dataset_omniglot'
params = torch.load(src .. '/params.t7')
config = {
    learningRate = params.lr,
    momentumDecay = 0.1,
    updateDecay = 0.01
}
require 'model'
setup(true, src)

function init()
  print("Network parameters:")
  print(params)
  reset_state(state)
  local epoch = 0
  local beginning_time = torch.tic()
  local start_time = torch.tic()
end

function run(mode)
  local dataset_loc = ""
  if mode == 'training' then
    dataset_loc = "/training/class"
  else
    dataset_loc = "/test/item"
  end

  for exp in paths.files("dataset/omniglot_oneshot") do
    if exp ~= "." and exp ~= ".." then
      for id = 1, 20 do 
        local fname = "dataset/omniglot_oneshot/" .. exp .. dataset_loc .. string.format("%02d", id) .. ".png"
        im = image.scale(image.load(fname), "32x32")
        im = torch.abs(im - 1)
        -- im[torch.le(im,0.5)] = 0
        -- im[torch.ge(im,0.5)] = 1

        local data = torch.zeros(params.bsize,1,32,32):cuda()
        data[1] = im
        local _, output = fp(data)
        local affines = torch.zeros(7*params.num_entities)
        for i = 0,params.num_entities-1 do
          local tmp = extract_node(model.rnns[1], 'affines_' .. (i+1)).data.module.output:double()[1]
          affines[{{(i*7+1),i*7+6}}] = tmp
          affines[{{(i*7+1)+6,(i*7+1)+6}}] = extract_node(model.rnns[1], 'intensity_' .. (i+1)).data.module.output:double()[1]
        end
        matio.save("dataset/omniglot_oneshot/" .. exp .. dataset_loc .. string.format("%02d", id) .. ".mat", {aff = affines})
      end
    end
  end
end

MAX_IMAGES_TO_DISPLAY = 30
plot = true
if plot then
  max_num = 1
end
init()
run('training')
run('test')