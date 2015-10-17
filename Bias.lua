local Bias, parent = torch.class('nn.Bias', 'nn.Module')

function Bias:__init(bsize, outputSize)
  parent.__init(self)
  self.output = torch.Tensor(bsize, outputSize)
  self.bsize = bsize
  local tmp1 = torch.rand(1,outputSize)
  self.bias = torch.repeatTensor(tmp1, bsize, 1)--torch.rand(bsize, outputSize)
  self.gradBias = torch.zeros(bsize, outputSize)
end

function Bias:updateOutput(input)
  self.output:copy(self.bias)
  return self.output
end

function Bias:updateGradInput(input, gradOutput)
  self.gradInput = torch.zeros(input:size()):cuda()
  -- self.gradBias:add(1, gradOutput)
  self.gradBias = torch.sum(gradOutput, 1)
  local gradBias_rep = torch.repeatTensor(self.gradBias, self.bsize, 1)
  self.bias:add(-5e3, gradBias_rep)
  return self.gradInput
end
