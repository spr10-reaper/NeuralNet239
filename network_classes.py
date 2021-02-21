import numpy as np

class LSTM:
	

	def __init__(self, inputSize, outputSize):
		
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

	def forgetGate(self, inputData):
	 	
		inputForget = np.matmul(self.weightInputForget, inputData)
		previousForget = np.matmul(self.weightPreviousForget, self.previousOutput)
		total = inputForget + previousForget + self.activationBiasForget

		self.f = sigmoid(total)
		return
	def inputGate(self, inputData):

		inputInput = np.matmul(self.weightInputInput, inputData)
		previousInput = np.matmul(self.weightPreviousInput, self.previousOutput)
		total = inputInput + previousInput + self.activationBiasInput

		self.i = sigmoid(total)
		return

	def inputNode(self, inputData):

		inputNode = np.matmul(self.weightInputNode, inputData)
		previousNode = np.matmul(self.weightPreviousNode, self.previousOutput)
		total = inputNode + previousNode + self.activationBiasNode

		self.i = sigmoid(total)
		return

	def outputGate(self, inputData):

		inputOutput = np.matmul(self.weightInputOutput, inputData)
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

		return self.currentOutput

#----------------end functions for calculating------------------#


class NormalMultivariate:

	def __init__(self, mean, std):

		self.meanVec = mean
		self.covMat = std

	def sample(self):

		sample = np.random.Generator.normal(size = self.meanVec.shape)
		fullSample = self.meanVec + self.covMat * sample
		covMatDet = np.dot(self.covMat, self.covMat)
		dimensions = len(self.covMat)

		quadratic = np.matmul(np.transpose(fullSample - self.meanVec), np.matmul(np.eye(dimensions) * covMat, (fullSample - self.meanVec)))

		logLikelihood = -0.5 * (np.log(covMatDet) + quadratic + dimensions * np.log(2 * np.pi))
		
		return fullSample, logLikelihood

	def entropy(self):
		
		covMatDet = np.dot(self.covMat, self.covMat)
		dimensions = len(self.covMat)
		entropy = dimensions * ( 1 + np.log(2 * np.pi)) / 2 + np.log(covMatDet) / 2


		return entropy
		

		


	
	
class Model:

	def __init__(self, numberOfNodes, inputData):
		
		self.actionLayer = [LSTM(numberOfNodes[i], numberOfNodes[i+1]) for i in range(len(numberOfNodes) - 1)])
		
		self.valueLayer = [LSTM(numberOfNodes[i], numberOfNodes[i+1]) for i in range(len(numberOfNodes) - 1)])
		self.actionLayer.append(LSTM(numberOfNodes[-1], outputSize))
		self.valueLayer.append(LSTM(numberOfNodes[-1], outputSize))


		self.actions = []
		self.states = []
		self.logProbs = []
		self.stateValues = []
		self.rewards = []


	def clearMemory(self):
		del self.actions[:]
		del self.states[:]
		del self.logProbs[:]
		del self.stateValues[:]
		del self.rewards[:]
		

	def forwardPropagate(self, inputData):

		data = inputData
		for module in self.valueLayer:
			
			module.inputData = data
			data = module.returnOutput()

		return np.arctanh(data)

	
	def actionPropagate(self, inputData):
		
		data = inputData

		for module in self.actionLayer:
			
			module.inputData = data
			data = module.returnOutput()

		return np.arctanh(data)

	

	def fullForward(self, inputData, action=None, evaluate=False):

		stateFull = self.forwardPropagate(inputData)
		stateValue = stateFull[:len(stateFull)]
		stateSTD = stateFull[len(stateFull):]

		actionFull = self.actionPropagate(inputData)
		actionValue = actionFull[:len(actionFull)]
		actionSTD = actionFull[len(actionFull):]
		
		actionDistribution = NormalMultivariate(actionMean, actionSTD)
		
		action, logProb = actionDistribution.sample()

		self.actions.append(action)
		self.logProbs.append(logProb)
		self.stateValues.append(stateValue)

		return actionDistribution.entropy()
		if not evaluate:
			


			
