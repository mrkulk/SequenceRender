-- Usage: th main.lua


require 'nn'
package.path = package.path .. ';modules/Affine/?.lua'

require 'INTM'
require 'ACR'
require 'Bias'
require 'optim'
require 'image'

twidth = 10
imwidth = 100
acr = nn.ACR(1,imwidth)
intm = nn.INTM(1,7,10, imwidth)
bias = nn.Bias(1, twidth*twidth)


net = nn.Sequential()
acr_wrapper = nn.Sequential()
acr_wrapper:add(nn.Replicate(2))
acr_wrapper:add(nn.SplitTable(1))

acr_in = nn.ParallelTable()
acr_in:add(bias)
acr_in:add(intm)

acr_wrapper:add(acr_in)
net:add(acr_wrapper)
net:add(acr)

data = torch.Tensor({0, 0 , 0.5, 0.5, 0, 0, 1})
data = data:reshape(1,7)
targets = torch.zeros(imwidth,imwidth):reshape(1,imwidth,imwidth)
outputs = net:forward(data):clone()
image.save('test.png', outputs[1])