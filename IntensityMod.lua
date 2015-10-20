local IntensityMod, parent = torch.class('nn.IntensityMod', 'nn.Module')

function IntensityMod:__init(...)
 parent.__init(self)
end

-- input format : {intensities, images}
function IntensityMod:updateOutput(input)
	-- print(input[1]:min(), input[2]:min())
	self.output = input[2]:clone()
	for i=1,input[1]:size(1) do 
		self.output[i] = self.output[i] * input[1][i][1]
	end 
	return self.output
end

function IntensityMod:updateGradInput(input, gradOutput)
	-- print(gradOutput:sum())
	-- print(input[1])
	self.gradInput = {torch.Tensor(input[1]:size()):zero():cuda(), torch.Tensor(input[2]:size()):zero():cuda()} 
	for i=1,input[1]:size(1) do 
		self.gradInput[1][i] = torch.cmul(input[2][i], gradOutput[i]):sum()
	-- 	-- print(gradOutput[i]:size(), self.gradInput[2][i]:size(), input[1][i][1])
	-- 	print(gradOutput[i])
		self.gradInput[2][i] = gradOutput[i] * input[1][i][1]
	end
	-- self.gradInput[1] = self.gradInput[1]:cuda()
	-- self.gradInput[2] = self.gradInput[2]:cuda()
	return self.gradInput
end
