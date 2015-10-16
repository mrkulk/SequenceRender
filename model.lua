-- require 'Normalize'
-- require 'componentMul'
-- require 'PowTable'
require 'nngraph'
require 'cunn'
require 'cudnn'
require 'rmsprop'
require 'stn'
require 'GradScale'
require 'PrintModule'
-- require 'ParallelParallel'
require 'IntensityMod'

model = {}

function transfer_data(x)
  return x:cuda()
end


function lstm(x, prev_c, prev_h)
  -- Calculate all four gates in one go
  local i2h = nn.Linear(params.rnn_size, 4*params.rnn_size)(x)
  local h2h = nn.Linear(params.rnn_size, 4*params.rnn_size)(prev_h)
  local gates = nn.CAddTable()({i2h, h2h})
  
  -- Reshape to (bsize, n_gates, hid_size)
  -- Then slize the n_gates dimension, i.e dimension 2
  local reshaped_gates =  nn.Reshape(4,params.rnn_size)(gates)
  local sliced_gates = nn.SplitTable(2)(reshaped_gates)
  
  -- Use select gate to fetch each gate and apply nonlinearity
  local in_gate          = nn.Sigmoid()(nn.SelectTable(1)(sliced_gates))
  local in_transform     = nn.Tanh()(nn.SelectTable(2)(sliced_gates))
  local forget_gate      = nn.Sigmoid()(nn.SelectTable(3)(sliced_gates))
  local out_gate         = nn.Sigmoid()(nn.SelectTable(4)(sliced_gates))

  local next_c           = nn.CAddTable()({
      nn.CMulTable()({forget_gate, prev_c}),
      nn.CMulTable()({in_gate,     in_transform})
  })
  local next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

  return next_c, next_h
end


-- function create_decoder(bsize, num_acrs, template_width, image_width)
--   local mod = nn.Sequential()
--   local decoder = nn.Parallel(2,2)
--   for ii=1,num_acrs do
--     local acr_wrapper = nn.Sequential()
--     acr_wrapper:add(nn.Replicate(2))

--     acr_wrapper:add(nn.SplitTable(1))

--     local acr_in = nn.ParallelTable()
--     local biasWrapper = nn.Sequential()
--       biasWrapper:add(nn.Bias(bsize, template_width*template_width))
--       --biasWrapper:add(nn.PrintModule("PostBias"))
--       biasWrapper:add(nn.Exp())
--       biasWrapper:add(nn.AddConstant(1))
--       biasWrapper:add(nn.Log())
--     acr_in:add(biasWrapper)

--     local INTMWrapper = nn.Sequential()
--       local splitter = nn.Parallel(2,2)

--         --tx
--         local tw = nn.Sequential()
--           tw:add(nn.Sigmoid())
--           tw:add(nn.AddConstant(-0.5))
--           tw:add(nn.MulConstant(image_width))
--           tw:add(nn.Reshape(bsize, 1))
--         splitter:add(tw)

--         --tx
--         local tw = nn.Sequential()
--           tw:add(nn.Sigmoid())
--           tw:add(nn.AddConstant(-0.5))
--           tw:add(nn.MulConstant(image_width))
--           tw:add(nn.Reshape(bsize, 1))
--         splitter:add(tw)

--         --sx
--         local tw = nn.Sequential()
--           -- tw:add(nn.Sigmoid())
--           tw:add(nn.Reshape(bsize, 1))
--         splitter:add(tw)
--         --sy
--         local tw = nn.Sequential()
--           -- tw:add(nn.Sigmoid())
--           tw:add(nn.Reshape(bsize, 1))
--         splitter:add(tw)

--         local tw = nn.Sequential()
--           tw:add(nn.Reshape(bsize, 1))
--         splitter:add(tw)

--         local tw = nn.Sequential()
--           tw:add(nn.Reshape(bsize, 1))
--         splitter:add(tw)


--         local intensityWrapper = nn.Sequential()
--           intensityWrapper:add(nn.Exp())
--           intensityWrapper:add(nn.AddConstant(1))
--           intensityWrapper:add(nn.Log())
--           intensityWrapper:add(nn.Reshape(bsize, 1))
--         splitter:add(intensityWrapper)

--       INTMWrapper:add(splitter)
--       INTMWrapper:add(nn.INTM(bsize, 7, 10, image_width))
--       -- INTMWrapper:add(nn.INTMReg())
--     acr_in:add(INTMWrapper)
--     acr_wrapper:add(acr_in)
--     acr_wrapper:add(nn.ACR(bsize, image_width))

--     decoder:add(acr_wrapper)
--   end
--   mod:add(decoder)

--   mod:add(nn.Reshape(num_acrs, image_width,image_width))
--   mod:add(nn.Sum(2))
--   mod:add(nn.Reshape(1,image_width,image_width))
--   mod:add(nn.Sigmoid())
--   return mod
-- end

function get_transformer(params)
  local x = nn.Identity()()
  local encoder_out = nn.Identity()()

  local outLayer = nn.Linear(params.rnn_size,6)(encoder_out)
  if true then
    outLayer.data.module.weight:fill(0)
    local bias = torch.FloatTensor(6):fill(0)
    bias[1]= 1-- 1+torch.rand(1)[1]*2
    bias[5]= 1-- 1+torch.rand(1)[1]*2
    -- bias[3]=torch.rand(1)[1]*2
    -- bias[6]=torch.rand(1)[1]*2
    outLayer.data.module.bias:copy(bias)
  end

  -- there we generate the grids
  -- local affines = nn.PrintModule("AFFINE:")(outLayer)
  local grid = nn.AffineGridGeneratorBHWD(32,32)(nn.View(2,3)(outLayer))

  -- first branch is there to transpose inputs to BHWD, for the bilinear sampler
  local tranet=nn.Transpose({2,3},{3,4})(x)

  local spanet = nn.BilinearSamplerBHWD()({tranet, grid})
  local sp_out = nn.Transpose({3,4},{2,3})(spanet)
  return nn.gModule({x, encoder_out}, {sp_out})
end


function create_network(params)
  local prev_s = nn.Identity()() -- LSTM
  local x = nn.Identity()() --input
  -- local prev_canvas = nn.Identity()()

  local num_acrs = params.num_acrs
  local template_width = params.template_width
  local image_width = params.image_width
  local bsize = params.bsize

  --- encoder ---
  local input_image = x-- nn.JoinTable(2)({x,prev_canvas})
  -- local enc1 = cudnn.SpatialMaxPooling(2,2)(nn.ReLU()(cudnn.SpatialConvolution(1, 32, 3, 3)(input_image)))
  -- local enc2 = cudnn.SpatialMaxPooling(2,2)(nn.ReLU()(cudnn.SpatialConvolution(32, 64, 3, 3)(enc1)))
  -- local fc1 = nn.Linear(64*6*6,params.rnn_size)((nn.Reshape(64*6*6)(enc2)))

  local enc1 = nn.Tanh()(nn.Linear(1024,2048)(nn.Reshape(1024)(x)))
  local fc1 = nn.Tanh()(nn.Linear(2048, params.rnn_size)(enc1))
  -- local fc1 = nn.Linear(512, params.rnn_size)(enc2)

  
  local rnn_i                = {[0] = nn.Identity()(fc1)}
  local next_s           = {}
  local split         = {prev_s:split(2 * params.layers)}
  for layer_idx = 1, params.layers do
    local prev_c         = split[2 * layer_idx - 1]
    local prev_h         = split[2 * layer_idx]
    local dropped        = nn.Dropout(params.dropout)(rnn_i[layer_idx - 1])
    local next_c, next_h = lstm(dropped, prev_c, prev_h)
    table.insert(next_s, next_c)
    table.insert(next_s, next_h)
    rnn_i[layer_idx] = next_h
  end

  local sts = {}
  local canvas = {}
  for i=1,params.num_acrs do
    sts[i] = {}
    local mem = nn.SpatialUpSamplingNearest(4)(nn.Reshape(1,template_width,template_width)(nn.Bias(bsize, template_width*template_width)(x)))
    local mem_out = mem--nn.Sigmoid()(mem)
    --intensity for each mem 

    local intensity = nn.Log()(nn.AddConstant(1)(nn.Exp()(nn.Linear(params.rnn_size, 1)(rnn_i[params.layers]))))--nn.Sigmoid()(nn.Linear(params.rnn_size, 1)(rnn_i[params.layers]))
    local mem_intensity = nn.IntensityMod()({intensity, mem_out})

    sts[i]["enc_out"] = nn.Identity()(rnn_i[params.layers])
    sts[i]["transformer"] = get_transformer(params)({mem_intensity, sts[i]["enc_out"]})
    -- adding up all frames on single canvas
    table.insert(canvas, sts[i]["transformer"])
  end

  local canvas_out = nn.Sigmoid()(nn.Reshape(1,image_width,image_width)(nn.Sum(2)(nn.JoinTable(2)(canvas))))

  local err = nn.MSECriterion()({canvas_out, x})
  return nn.gModule({x,prev_s}, {err, nn.Identity()(next_s), canvas_out})
end



function setup()
  print("Creating a RNN LSTM network.")
  local core_network = create_network(params)
  core_network:cuda()
  paramx, paramdx = core_network:getParameters()
  model.s = {}
  model.ds = {}
  model.start_s = {}
  for j = 0, params.seq_length do
    model.s[j] = {}
    for d = 1, 2 * params.layers do
      model.s[j][d] = transfer_data(torch.zeros(params.bsize, params.rnn_size))
    end
  end
  for d = 1, 2 * params.layers do
    model.start_s[d] = transfer_data(torch.zeros(params.bsize, params.rnn_size))
    model.ds[d] = transfer_data(torch.zeros(params.bsize, params.rnn_size))
  end
  model.core_network = core_network
  model.rnns = g_cloneManyTimes(core_network, params.seq_length)
  model.norm_dw = 0
  model.err = transfer_data(torch.zeros(params.seq_length))
end



function reset_state()
  for j = 0, params.seq_length do
    for d = 1, 2 * params.layers do
      model.s[j][d]:zero()
    end
  end
end

function reset_ds()
  for d = 1, #model.ds do
    model.ds[d]:zero()
  end
end

function fp(data)
  g_replace_table(model.s[0], model.start_s)
  reset_state()
  local next_canvas
  for i = 1, params.seq_length do
    local x = data:clone()
    -- local prev_x 
    -- if i == 1 then
    --   prev_x = torch.zeros(params.bsize,1,32,32)
    -- else
    --   prev_x = next_x:clone()
    --   prev_x = torch.reshape(prev_x, params.bsize, 1, 32,32)
    -- end
    local s = model.s[i - 1]
    model.err[i], model.s[i], new_canvas = unpack(model.rnns[i]:forward({x, s}))
  end
  g_replace_table(model.start_s, model.s[params.seq_length])
  return model.err:mean(), new_canvas
end

function bp(data)
  paramdx:zero()
  reset_ds()
  for i = params.seq_length, 1, -1 do
    local x = data:clone()
    -- local prev_x 
    -- if i == 1 then
    --   prev_x = torch.zeros(params.bsize,1,32,32)
    -- else
    --   prev_x = next_x:clone()
    -- end
    local s = model.s[i - 1]
    local derr = transfer_data(torch.ones(1))
    local dnewx = transfer_data(torch.zeros(params.bsize, 32, 32))
    local tmp = model.rnns[i]:backward({x, s},
                                       {derr, model.ds, dnewx})
    g_replace_table(model.ds, tmp[2])

    cutorch.synchronize()
  end
  model.norm_dw = paramdx:norm()
  if model.norm_dw > params.max_grad_norm then
    local shrink_factor = params.max_grad_norm / model.norm_dw
    paramdx:mul(shrink_factor)
  end

  paramx = rmsprop(paramdx, paramx, config, state)

  -- paramx:add(paramdx:mul(-params.lr))
end


-- function run_valid()
--   reset_state(state_valid)
--   g_disable_dropout(model.rnns)
--   local len = (state_valid.data:size(1) - 1) / (params.seq_length)
--   local perp = 0
--   for i = 1, len do
--     perp = perp + fp(state_valid)
--   end
--   print("Validation set perplexity : " .. g_f3(torch.exp(perp / len)))
--   g_enable_dropout(model.rnns)
-- end

-- local function run_test()
--   reset_state(state_test)
--   g_disable_dropout(model.rnns)
--   local perp = 0
--   local len = state_test.data:size(1)
--   g_replace_table(model.s[0], model.start_s)
--   for i = 1, (len - 1) do
--     local x = state_test.data[i]
--     local y = state_test.data[i + 1]
--     perp_tmp, model.s[1] = unpack(model.rnns[1]:forward({x, y, model.s[0]}))
--     perp = perp + perp_tmp[1]
--     g_replace_table(model.s[0], model.s[1])
--   end
--   print("Test set perplexity : " .. g_f3(torch.exp(perp / (len - 1))))
--   g_enable_dropout(model.rnns)
-- end
