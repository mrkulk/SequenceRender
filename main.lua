-- Usage: th main.lua


package.path = package.path .. ';modules/Affine/?.lua'
require 'nn'
require 'INTM'
require 'ACR'
require 'Bias'
require 'optim'
require 'image'
require 'dataset-mnist'
require 'model'
require 'cutorch'

params = {
	bsize = 3,
	image_width = 32,
	template_width = 10,
	num_acrs = 1,
	rnn_size = 200,
	seq_length=20,
	layers=2,
	decay=2,
	dropout=0,
	init_weight=0.1,
	lr=1e-2,
	max_steps=20000,
	max_grad_norm=5
}

-- torch.manualSeed(1)

-- create training set and normalize
trainData = mnist.loadTrainSet(nbTrainingPatches, geometry)
trainData:normalizeGlobal(mean, std)

-- create test set and normalize
testData = mnist.loadTestSet(nbTestingPatches, geometry)
testData:normalizeGlobal(mean, std)


model = create_network(params)
model:cuda()
ret = model:forward({torch.rand(params.bsize, 1, 32, 32):cuda(), torch.rand(params.bsize, 1, 32, 32):cuda(),
    {torch.zeros(params.bsize, params.rnn_size):cuda(), torch.zeros(params.bsize, params.rnn_size):cuda(), torch.zeros(params.bsize, params.rnn_size):cuda(), torch.zeros(params.bsize, params.rnn_size):cuda() },
	})

-- image.save('test.png',ret[1])