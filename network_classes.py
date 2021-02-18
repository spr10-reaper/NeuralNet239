import numpy as np

class LSTM:
	

	def __init__(self, inputSize, outputSize, inputData):
		
		self.inputData = inputData

		self.weightInputInput = np.random((outputSize, inputSize))
		self.weightPrevious = np.random((outputSize, outputSize))
		self.activationBiasInput = np.random(outputSize)

		self.weightInputNode = np.random((outputSize, inputSize))
		self.weightPreviousNode = np.random((outputSize, outputSize))
		self.activationBiasNode = np.random(outputSize)
		
		self.weightInputForget = np.random((outputSize, inputSize))
		self.weightPreviousForget = np.random((outputSize, outputSize))
		self.activationBiasForget = np.random(outputSize)

		self.weightInputOutput = np.random((outputSize, inputSize))
		self.weightPreviousOutput = np.random((outputSize, outputSize))
		self.activationBiasOutput = np.random(outputSize)

		self.previousState = np.random(outputSize)
		self.previousOutput = np.random(outputSize)
		self.currentState = np.random(outputSize)
		self.currentOutput = np.zeros(outputSize)

		self.f = np.zeros(outputSize)
		self.i = np.zeros(outputSize)
		self.g = np.zeros(outputSize)
		self.o = np.zeros(outputSize)

		return

#-------------------Getters and Setters-------------------------#

	@property
	def inputData(self):

		return self.inputData

	@inputData.setter
	def inputData(self, data):
		
		self.inputData = data
		return

	@property
	def weightInputInput(self):

		return self.weightInputInput

	@weightInputInput.setter
	def weightInputInput(self, data):
		
		self.weightInputInput = data

		return

	@property
	def weightPrevious(self):

		return self.weightPrevious
	
	@weightPrevious.setter
	def weightPrevious(self, x):
		
		self.weightPrevious = x

		return

	@property
	def activationBiasInput(self):

		return self.activationBiasInput


	@activationBiasInput.setter
	def activationBiasInput(self, x):
		
		self.activationBiasInput = x

		return


	@property
	def weightInputNode(self):
		
		return self.weightInputNode

	@weightInputNode.setter
	def weightInputNode(self, x):
		
		self.weightInputNode = x
		return

	@property
	def weightPreviousNode(self):

		return self.weightPreviousNode
	
	@weightPreviousNode.setter
	def weightPreviousNode(self, x):
		
		self.weightPreviousNode = x
		return

	@property
	def activationBiasNode(self):
		
		return self.activationBiasNode
	
	@activationBiasNode.setter
	def activationBiasNode(self, x):
		
		self.activationBiasNode = x
		return
	
	@property
	def weightInputForget(self):
		
		return self.weightInputForget
	
	@weightInputForget.setter
	def weightInputForget(self, x):
		
		self.weightInputForget = x
		return

	@property
	def weightPreviousForget(self):
		
		return self.weightPreviousForget
	
	@weightPreviousForget.setter
	def weightPreviousForget(self, x):
		
		self.weightPreviousForget = x
		return

	@property
	def activationBiasForget(self):
		
		return self.activationBiasForget
	
	@activationBiasForget.setter
	def activationBiasForget(self, x):
		
		self.activationBiasForget = x
		return

	@property
	def weightInputOutput(self):
		
		return self.weightInputOutput
	
	@weightInputOutput.setter
	def weightInputOutput(self, x):
		
		self.weightInputOutput = x
		return

	@property
	def weightPreviousOutput(self):
		
		return self.weightPreviousOutput
	
	@weightPreviousOutput.setter
	def weightPreviousOutput(self, x):
		
		self.weightPreviousOutput = x
		return

	@property
	def activationBiasOutput(self):
		
		return self.activationBiasOutput
	
	@activationBiasOutput.setter
	def activationBiasOutput(self, x):
		
		self.activationBiasOutput = x
		return

	@property
	def previousState(self):
		
		return self.previousState
	
	@previousState.setter
	def previousState(self, x):
		
		self.previousState = x
		return

	@property
	def previousOutput(self):
		
		return self.previousOutput
	
	@previousOutput.setter
	def previousOutput(self, x):
		
		self.previousOutput = x
		return

	@property
	def currentState(self):
		
		return self.currentState
	
	@currentState.setter
	def currentState(self, x):
		
		self.currentState = x
		return

	@property
	def currentOutput(self):
		
		return self.currentOutput
	
	@currentOutput.setter
	def currentOutput(self, x):
		
		self.currentOutput = x
		return

`
#--------------------end of getters and setters-----------------#


#--------------------functions for calculating------------------#

	def forgetGate(self):
	 	
		inputForget = np.matmul(self.weightInputForget, self.inputData)
		previousForget = np.matmul(self.weightPreviousForget, self.previousOutput)
		total = inputForget + previousForget + self.activationBiasForget

		self.f = sigmoid(total)
		return
	def inputGate(self):

		inputInput = np.matmul(self.weightInputInput, self.inputData)
		previousInput = np.matmul(self.weightPreviousInput, self.previousOutput)
		total = inputInput + previousInput + self.activationBiasInput

		self.i = sigmoid(total)
		return

	def inputNode(self):

		inputNode = np.matmul(self.weightInputNode, self.inputData)
		previousNode = np.matmul(self.weightPreviousNode, self.previousOutput)
		total = inputNode + previousNode + self.activationBiasNode

		self.i = sigmoid(total)
		return

	def outputGate(self):

		inputOutput = np.matmul(self.weightInputOutput, self.inputData)
		previousOutput = np.matmul(self.weightPreviousOutput, self.previousOutput)
		total = inputOutput + previousOutput + self.activationBiasOutput

		self.o = sigmoid(total)
		return

		

	def forgetCombo(self):
		
		self.combinedForget = self.f * self.previousState

		return

	def makeState(self):
		
		self.currentState = self.i + self.g + self.combinedForget
		self.previousState = self.currentState

		return

	def returnOutput(self):

		self.currentOutput = self.o * np.tanh(self.currentState)

		return

#----------------end functions for calculating------------------#
	
	
class Layer:

	def __init__(self, numberOfNodes, ):


		
