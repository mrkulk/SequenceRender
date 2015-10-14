-- require 'Normalize'
-- require 'componentMul'
-- require 'GradScale'
-- require 'PowTable'
require 'nngraph'
require 'cunn'
require 'cudnn'
-- require 'ParallelParallel'

function transfer_data(x)
  return x:cuda()
end


function lstm(x, prev_c, prev_h)
  -- Calculate all four gates in one go
  local i2h = nn.Linear(params.rnn_size, 4*params.rnn_size)(x)
  local h2h = nn.Linear(params.rnn_size, 4*params.rnn_size)(prev_h)
  local gates = nn.CAddTable()({i2h, h2h})
  
  -- Reshape to (batch_size, n_gates, hid_size)
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


function create_decoder(bsize, num_acrs, template_width, image_width)
  local mod = nn.Sequential()
  local decoder = nn.Parallel(2,2)
  for ii=1,num_acrs do
    local acr_wrapper = nn.Sequential()
    acr_wrapper:add(nn.Replicate(2))

    acr_wrapper:add(nn.SplitTable(1))

    local acr_in = nn.ParallelTable()
    local biasWrapper = nn.Sequential()
      biasWrapper:add(nn.Bias(bsize, template_width*template_width))
      --biasWrapper:add(nn.PrintModule("PostBias"))
      biasWrapper:add(nn.Exp())
      biasWrapper:add(nn.AddConstant(1))
      biasWrapper:add(nn.Log())
    acr_in:add(biasWrapper)

    local INTMWrapper = nn.Sequential()
      local splitter = nn.Parallel(2,2)
        for i = 1,2 do
          splitter:add(nn.Reshape(bsize, 1))
        end

        --sx
        local tw = nn.Sequential()
          tw:add(nn.Exp())
          tw:add(nn.AddConstant(1))
          tw:add(nn.Log())
          tw:add(nn.Reshape(bsize, 1))
          -- tw:add(nn.PrintModule('INTENSITY MOD'))
        splitter:add(tw)
        --sy
        local tw = nn.Sequential()
          tw:add(nn.Exp())
          tw:add(nn.AddConstant(1))
          tw:add(nn.Log())
          tw:add(nn.Reshape(bsize, 1))
          -- tw:add(nn.PrintModule('INTENSITY MOD'))
        splitter:add(tw)

        local tw = nn.Sequential()
          -- tw:add(nn.Exp())
          -- tw:add(nn.AddConstant(1))
          -- tw:add(nn.Log())
          tw:add(nn.Reshape(bsize, 1))
          -- -- tw:add(nn.PrintModule('INTENSITY MOD'))
        splitter:add(tw)

        local tw = nn.Sequential()
        
          -- tw:add(nn.Exp())
          -- tw:add(nn.AddConstant(1))
          -- tw:add(nn.Log())
          tw:add(nn.Reshape(bsize, 1))
          -- tw:add(nn.PrintModule('INTENSITY MOD'))
        splitter:add(tw)


        local intensityWrapper = nn.Sequential()
          intensityWrapper:add(nn.Exp())
          intensityWrapper:add(nn.AddConstant(1))
          intensityWrapper:add(nn.Log())
          intensityWrapper:add(nn.Reshape(bsize, 1))
          -- intensityWrapper:add(nn.PrintModule('INTENSITY MOD'))
        splitter:add(intensityWrapper)
      INTMWrapper:add(splitter)
      INTMWrapper:add(nn.INTM(bsize, 7, 10, image_width))
      -- INTMWrapper:add(nn.INTMReg())
    acr_in:add(INTMWrapper)
    acr_wrapper:add(acr_in)
    acr_wrapper:add(nn.ACR(bsize, image_width))

    decoder:add(acr_wrapper)
  end
  mod:add(decoder)
  return mod
end

function create_network(params)
  local prev_s = nn.Identity()() -- LSTM
  local x = nn.Identity()() --input
  local prev_canvas = nn.Identity()()

  local num_acrs = params.num_acrs
  local template_width = params.template_width
  local image_width = params.image_width
  local bsize = params.bsize

  --- encoder ---
  local input_image = nn.JoinTable(2)({x,prev_canvas})
  local enc1 = cudnn.SpatialMaxPooling(2,2)(nn.ReLU()(cudnn.SpatialConvolution(2, 32, 3, 3)(input_image)))
  local enc2 = cudnn.SpatialMaxPooling(2,2)(nn.ReLU()(cudnn.SpatialConvolution(32, 64, 3, 3)(enc1)))
  -- local fc1 = nn.ReLU()(nn.Linear(64*6*6,200)((nn.Reshape(64*6*6)(enc2))))
  -- local encout = nn.Reshape(num_acrs,7)(nn.Linear(200, num_acrs*7)(fc1))
  local fc1 = nn.Linear(64*6*6,params.rnn_size)((nn.Reshape(64*6*6)(enc2)))
  
  local i                = {[0] = nn.Identity()(fc1)}
  local next_s           = {}
  local split         = {prev_s:split(2 * params.layers)}
  for layer_idx = 1, params.layers do
    local prev_c         = split[2 * layer_idx - 1]
    local prev_h         = split[2 * layer_idx]
    local dropped        = nn.Dropout(params.dropout)(i[layer_idx - 1])
    local next_c, next_h = lstm(dropped, prev_c, prev_h)
    table.insert(next_s, next_c)
    table.insert(next_s, next_h)
    i[layer_idx] = next_h
  end
  local encout = nn.Reshape(num_acrs,7)(nn.Linear(params.rnn_size, num_acrs*7)(i[params.layers]))

  decoder = create_decoder(bsize, num_acrs, template_width, image_width)(encout)

  local mod = nn.gModule({x,prev_canvas, prev_s},{decoder})
  return mod
  -- local MEM              = nn.Identity()() -- memory
  -- local prev_read_key    = nn.Identity()() -- " "
  -- local prev_read_val    = nn.Identity()() 
  -- local prev_write_key   = nn.Identity()()
  -- local prev_write_val   = nn.Identity()()
  -- local prev_write_erase   = nn.Identity()()

  -- -- targets whenever available (specified fragments of the program)
  -- local true_read_key    = nn.Identity()()
  -- local true_read_val    = nn.Identity()() 
  -- local true_write_key   = nn.Identity()()
  -- local true_write_val   = nn.Identity()()
  -- local true_write_erase = nn.Identity()()

  -- local head_dim = params.rows * params.cols
  -- local reshape_prev_read_key    = nn.Reshape(params.batch_size, head_dim)(prev_read_key)
  -- local reshape_prev_read_val    = nn.Reshape(params.batch_size, head_dim)(prev_read_val)
  -- local reshape_prev_write_key   = nn.Reshape(params.batch_size, head_dim)(prev_write_key)
  -- local reshape_prev_write_val   = nn.Reshape(params.batch_size, head_dim)(prev_write_val)
  -- local reshape_prev_write_erase = nn.Reshape(params.batch_size, head_dim)(prev_write_erase)

  -- local concat_x = nn.JoinTable(2)({reshape_prev_read_key, reshape_prev_read_val, reshape_prev_write_key , reshape_prev_write_val, reshape_prev_write_erase })
  -- local remapped_x = nn.Linear(5*head_dim, params.rnn_size)(concat_x)

  -- local i                = {[0] = nn.Identity()(remapped_x)}

  -- local next_s           = {}
  -- local split         = {prev_s:split(2 * params.layers)}
  -- for layer_idx = 1, params.layers do
  --   local prev_c         = split[2 * layer_idx - 1]
  --   local prev_h         = split[2 * layer_idx]
  --   local dropped        = nn.Dropout(params.dropout)(i[layer_idx - 1])
  --   local next_c, next_h = lstm(dropped, prev_c, prev_h)
  --   table.insert(next_s, next_c)
  --   table.insert(next_s, next_h)
  --   i[layer_idx] = next_h
  -- end

  -- -- local h2y              = nn.Linear(params.rnn_size, params.input_dim)
  -- -- local dropped          = nn.Dropout(params.dropout)(i[params.layers])
  -- -- local pred             = nn.LogSoftMax()(h2y(dropped))

  -- ---------------------- Memory Ops ------------------
  -- local read_channel = nn.ReLU()(i[params.layers])--nn.ReLU()(nn.Linear(params.rnn_size,params.rnn_size)(i[params.layers]))
  -- -- local read_key = nn.Reshape(params.rows,params.cols)(nn.SoftMax()(nn.Sigmoid()(nn.Linear(params.rnn_size, params.rows*params.cols)(read_channel))))
  -- local read_key = nn.Reshape(params.rows,params.cols)(nn.Sigmoid()(nn.Linear(params.rnn_size, params.rows*params.cols)(read_channel)))
  -- -- local read_key = nn.Power(1.5)(read_key1)
  -- local read_val = nn.componentMul()({MEM, read_key})

  -- local write_channel = nn.ReLU()(i[params.layers])--nn.ReLU()(nn.Linear(params.rnn_size,params.rnn_size)(i[params.layers]))
  -- -- local write_key = nn.Reshape(params.rows,params.cols)(nn.SoftMax()(nn.Sigmoid()(nn.Linear(params.rnn_size, params.rows*params.cols)(write_channel))))
  -- local write_key = nn.Reshape(params.rows,params.cols)(nn.Sigmoid()(nn.Linear(params.rnn_size, params.rows*params.cols)(write_channel)))
  -- -- local write_key = nn.Power(1.5)(write_key1)
  -- local write_val = nn.Reshape(params.rows,params.cols)(nn.Linear(params.rnn_size, params.rows*params.cols)(write_channel)) 
  -- local write_erase = nn.Reshape(params.rows,params.cols)(nn.Linear(params.rnn_size, params.rows*params.cols)(write_channel)) 

  -- local erase_val_interim = nn.componentMul()({write_key, write_erase})
  -- local erase_val = nn.AddConstant(1)(nn.MulConstant(-1)(erase_val_interim))
  -- local erase_MEM = nn.componentMul()({MEM, erase_val})

  -- local add_val_interim = nn.componentMul()({write_key, write_val})
  -- local add_MEM = nn.CAddTable()({erase_MEM, add_val_interim})

  -- local err_rk = nn.MSECriterion()({read_key, nn.Reshape(params.rows * params.cols)(true_read_key)})
  -- local err_rv = nn.MSECriterion()({read_val, nn.Reshape(params.rows * params.cols)(true_read_val)})
  -- local err_wk = nn.MSECriterion()({write_key, nn.Reshape(params.rows * params.cols)(true_write_key)})
  -- local err_wv = nn.MSECriterion()({write_val, nn.Reshape(params.rows * params.cols)(true_write_val)})
  -- local err_we = nn.MSECriterion()({write_erase, nn.Reshape(params.rows * params.cols)(true_write_erase)})

  -- local module           = nn.gModule({prev_s, MEM, prev_read_key, prev_read_val, prev_write_key, prev_write_val, prev_write_erase,
  --                                                   true_read_key, true_read_val, true_write_key, true_write_val, true_write_erase},
  --                                     {err_rk, err_rv, err_wk, err_wv, err_we,
  --                                     nn.Identity()(next_s), add_MEM, read_key, read_val, write_key, write_val, write_erase})
  
  -- return module

end



function setup()
  print("Creating a RNN LSTM network.")
  local core_network = create_network()
  core_network:cuda()
  core_network:getParameters():uniform(-params.init_weight, params.init_weight)
  paramx, paramdx = core_network:getParameters()
  
  local model = {}
  model.s = {}
  model.ds = {}
  model.start_s = {}

  for j = 0, params.seq_length do
    model.s[j] = {}
    for d = 1, 2 * params.layers do
      model.s[j][d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
    end
  end

  for d = 1, 2 * params.layers do
    model.start_s[d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
    model.ds[d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
  end

  model.core_network = core_network
  model.rnns = g_cloneManyTimes(core_network, params.seq_length)
  model.norm_dw = 0
  model.err = transfer_data(torch.zeros(params.seq_length)) 

  return model
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

function fp(mode, state, target)
  local predicted_ret, target_ret --final return value (target and predicted)
  reset_state()

  for i = 1, params.seq_length do
    local ret

    if mode == "fake" then
      ret = model.rnns[i]:forward({
        model.s[i-1], model.MEM[i-1], model.read_key[i-1], model.read_val[i-1], model.write_key[i-1], model.write_val[i-1], model.write_erase[i-1],
        state.data.true_read_key[i], state.data.true_read_val[i], state.data.true_write_key[i], state.data.true_write_val[i], state.data.true_write_erase[i]
      })
    else

      local rk, rv, wk, wv, we
      if TARGET_CACHE[i].cmd == "write" then
        rk = torch.zeros(params.batch_size, params.rows, params.cols):cuda()
        rv = torch.zeros(params.batch_size, params.rows, params.cols):cuda()
        wk = TARGET_CACHE[i].key:clone(); wk = wk:cuda()
        wv = TARGET_CACHE[i].val:clone(); wv = wv:cuda()
        -- print(TARGET_CACHE[i].write_erase_cache:sum())
        we = TARGET_CACHE[i].write_erase_cache:clone();we = we:cuda()-- torch.zeros(params.batch_size, params.rows, params.cols):cuda()
      else
        rk = TARGET_CACHE[i].key:cuda()
        if TARGET_CACHE[i].mode == "return" then
          local map = TARGET_CACHE[i].map
          rv = torch.zeros(params.batch_size, params.rows, params.cols)
          rv[{{}, {map.from_row[1], map.from_row[2]}, {map.from_col[1], map.from_col[2]}}] = target:clone()
          rv = rv:cuda()
          -- rv = TARGET_CACHE[i].memory:cuda()
          target_ret = rv:clone(); target_ret = target_ret:float()
        else
          rv = torch.zeros(params.batch_size, params.rows, params.cols):cuda()
        end
        wk = torch.zeros(params.batch_size, params.rows, params.cols):cuda()
        wv = torch.zeros(params.batch_size, params.rows, params.cols):cuda()
        we = torch.zeros(params.batch_size, params.rows, params.cols):cuda()
      end

      ret = model.rnns[i]:forward({
        model.s[i-1], model.MEM[i-1], model.read_key[i-1], model.read_val[i-1], model.write_key[i-1], model.write_val[i-1], model.write_erase[i-1],
        rk, rv, wk, wv, we
      })    
      if TARGET_CACHE[i].mode == "return" then
        predicted_ret = ret[9]:float() --read_val
      end
    end

    model.err_rk[i] = ret[1][1]
    model.err_rv[i] = ret[2][1]
    model.err_wk[i] = ret[3][1]
    model.err_wv[i] = ret[4][1]
    model.err_we[i] = ret[5][1]
    g_replace_table(model.s[i], ret[6])
    model.MEM[i]:copy(ret[7]:clone()) 
    model.read_key[i]:copy(ret[8]:clone())
    model.read_val[i]:copy(ret[9]:clone())
    model.write_key[i]:copy(ret[10]:clone())
    model.write_val[i]:copy(ret[11]:clone())
    model.write_erase[i]:copy(ret[12]:clone())
  end
  -- g_replace_table(model.start_s, model.s[params.seq_length])
  return predicted_ret, target_ret, model.err_rk:mean() + model.err_rv:mean() + model.err_wk:mean() + model.err_wv:mean() + model.err_we:mean()
end

function bp(mode,state, target)
  paramdx:zero()
  reset_ds()
  for i = params.seq_length, 1, -1 do
    local derr_rk, derr_rv, derr_wk, derr_wv, derr_we

    local ret
    if mode == "fake" then
      derr_rk = transfer_data(torch.ones(1)); derr_rv = transfer_data(torch.ones(1))
      derr_wk = transfer_data(torch.ones(1)); derr_wv = transfer_data(torch.ones(1)); derr_we = transfer_data(torch.ones(1))

      ret = model.rnns[i]:backward(
      {
        model.s[i-1], model.MEM[i-1], model.read_key[i-1], model.read_val[i-1], model.write_key[i-1], model.write_val[i-1], model.write_erase[i-1],
        state.data.true_read_key[i], state.data.true_read_val[i], state.data.true_write_key[i], state.data.true_write_val[i], state.data.true_write_erase[i]
      },
      {
        derr_rk, derr_rv, derr_wk, derr_wv, derr_we,
        model.ds, model.ds_MEM, model.ds_read_key, model.ds_read_val, model.ds_write_key, model.ds_write_val, model.ds_write_erase
      })
    else

      local rk, rv, wk, wv, we
      if TARGET_CACHE[i].cmd == "write" then
        derr_rk = transfer_data(torch.ones(1)); derr_rv = transfer_data(torch.zeros(1))
        derr_wk = transfer_data(torch.ones(1)); derr_wv = transfer_data(torch.ones(1)); derr_we = transfer_data(torch.ones(1))
        rk = torch.zeros(params.batch_size, params.rows, params.cols):cuda()
        rv = torch.zeros(params.batch_size, params.rows, params.cols):cuda()
      
        wk = TARGET_CACHE[i].key:clone(); wk = wk:cuda()
        wv = TARGET_CACHE[i].val:clone(); wv = wv:cuda()
        we = TARGET_CACHE[i].write_erase_cache:clone();we = we:cuda()

      else
        derr_rk = transfer_data(torch.ones(1)); 
        derr_wk = transfer_data(torch.ones(1)); derr_wv = transfer_data(torch.zeros(1)); derr_we = transfer_data(torch.zeros(1))
        rk = TARGET_CACHE[i].key:clone(); rk = rk:cuda()
        if TARGET_CACHE[i].mode == "return" then
          -- rv = TARGET_CACHE[i].memory:cuda()
          local map = TARGET_CACHE[i].map
          rv = torch.zeros(params.batch_size, params.rows, params.cols)
          rv[{{}, {map.from_row[1], map.from_row[2]}, {map.from_col[1], map.from_col[2]}}] = target:clone()
          rv = rv:cuda()
          derr_rv = transfer_data(torch.ones(1)) --read output 
        else
          rv = torch.zeros(params.batch_size, params.rows, params.cols):cuda()
          derr_rv = transfer_data(torch.zeros(1)) --if we write properly, read will be good
        end
        wk = torch.zeros(params.batch_size, params.rows, params.cols):cuda()
        wv = torch.zeros(params.batch_size, params.rows, params.cols):cuda()
        we = torch.zeros(params.batch_size, params.rows, params.cols):cuda()
      end

      ret = model.rnns[i]:backward(
      {
        model.s[i-1], model.MEM[i-1], model.read_key[i-1], model.read_val[i-1], model.write_key[i-1], model.write_val[i-1], model.write_erase[i-1],
        rk, rv, wk, wv, we
      },
      {
        derr_rk, derr_rv, derr_wk, derr_wv, derr_we,
        model.ds, model.ds_MEM, model.ds_read_key, model.ds_read_val, model.ds_write_key, model.ds_write_val, model.ds_write_erase
      })  

      if TARGET_CACHE[i].mode == "return" then
          -- print(model.read_key[i][1])
          -- print(rk[1])
      end 
    end


    g_replace_table(model.ds, ret[1])
    model.ds_MEM:copy(ret[2])
    model.ds_read_key:copy(ret[3])
    model.ds_read_val:copy(ret[4])
    model.ds_write_key:copy(ret[5])
    model.ds_write_val:copy(ret[6])
    model.ds_write_erase:copy(ret[7])

    print(model.ds_read_key[1])
    -- print('------------')
    -- for ii=1,7 do
    --   print(ret[ii][1]:sum())
    -- end

    cutorch.synchronize()
  end

  if mode ~= "fake" then
    model.norm_dw = paramdx:norm()
    if model.norm_dw > params.max_grad_norm then
      local shrink_factor = params.max_grad_norm / model.norm_dw
      paramdx:mul(shrink_factor)
    end
    paramx:add(paramdx:mul(-params.lr))
  end
end


function eval(mode, state)
  reset_state()
  g_disable_dropout(model.rnns)
  local perp = 0
  for i = 1, params.seq_length do
    perp = perp + fp(state)
  end
  print(mode .. " set perplexity : " .. g_f3(torch.exp(perp / params.seq_length)))
  g_enable_dropout(model.rnns)
end
