-- require 'nn'
local PrintModule, parent = torch.class('nn.PrintModule', 'nn.Module')

function isnan(x) return x ~= x end

function PrintModule:__init(name)
  self.name = name
end

function PrintModule:updateOutput(input)
  -- if isnan(input) then
	  print(self.name.." input: ")
	  print(input)
	-- end
  self.output = input
  return input
end

function PrintModule:updateGradInput(input, gradOutput)
	-- if isnan(gradOutput) then
	  print(self.name.." gradInput:")
	  print(gradOutput)
	-- end
  self.gradInput = gradOutput
  return self.gradInput
end

