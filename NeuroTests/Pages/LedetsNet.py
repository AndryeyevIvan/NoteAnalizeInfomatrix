import random
import math

class nn:
    def __init__(self, netStruct = [], activationFunction = "sigmoid", learningRate = 0.1, netWeights = [], netBiases = [], correctAnswers = [], inputLayers = []):
        self.struct = []
        self.weights = []
        self.biases = []
        self.errors = []
        self.corrAnswers = []
        self.answers = []
        self.inLayers = []
        self.epochErrors = []
        self.actiFunc = ""
        self.lrnRate = 0
        self.StructInit(netStruct)
        self.actiFunc = activationFunction
        self.weights = netWeights
        self.biases = netBiases
        self.inLayers = inputLayers
        self.answers = correctAnswers
        if correctAnswers:
            self.corrAnswers = correctAnswers[0]
        self.lrnRate = learningRate

    def StructInit(self, netStruct = []):
        for i in netStruct:
            d = []
            for j in range(i):
                d += [0]
            self.struct += [d]
        # print(self.struct)

    def WeightsInit(self):
        for i in range(len(self.struct) - 1):
            c = []
            for j in range(len(self.struct[i + 1])):
                d = []
                for g in range(len(self.struct[i])):
                    d += [random.uniform(-0.5, 0.5)]
                c += [d]
            self.weights += [c]
        # print(self.weights)

    def BiasesInit(self, rand = False):
        for i in range(len(self.struct) - 1):
            d = []
            for j in range(len(self.struct[i + 1])):
                if rand == True:
                    d += [random.uniform(-0.5, 0.5)]
                else:
                    d += [0]
            self.biases.append(d)
        # print(self.biases)

    def Activation(self, x):
        if self.actiFunc == "sigmoid":
            return float(1 / (1 + math.exp(-x)))
        elif self.actiFunc == "relu":
            return max(0, x)
        else:
            return x
            print("Without actifunc")

    def ForwardPropagation(self):
        for i in range(len(self.struct) - 1):
            for j in range(len(self.struct[i + 1])):
                d = 0
                for g in range(len(self.struct[i])):
                    d += self.struct[i][g] * self.weights[i][j][g]
                self.struct[i + 1][j] = self.Activation(d + self.biases[i][j])
        # print(self.struct)

    def ErrorsInit(self):
        self.errors = []
        for i in range(1, len(self.struct)):
            d = []
            for j in range(len(self.struct[i])):
                d += [0]
            self.errors.append(d)
        # print(self.errors)

    def OutputErrors(self):
        d = []
        for i in range(len(self.struct[-1])):
            d += [(self.struct[-1][i] - self.corrAnswers[i]) * self.struct[-1][i] * (1 - self.struct[-1][i])]
        self.errors[len(self.struct) - 2] = d
        # print(self.errors)

    def HiddenErrors(self):
        for i in range(1, len(self.struct) - 1):
            index = len(self.struct) - 1 - i
            for j in range(len(self.struct[index])):
                self.errors[index - 1][j] = 0
            for j in range(len(self.struct[index + 1])):
                for g in range(len(self.struct[index])):
                    self.errors[index - 1][g] += self.weights[index][j][g] * self.errors[index][j]
        for i in range(len(self.errors) - 1):
            for j in range(len(self.errors[i])):
                self.errors[i][j] = self.errors[i][j] * self.struct[i + 1][j] * (1 - self.struct[i + 1][j])
        # print(self.errors)

    def UpdateWeightsAndBiases(self):
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                for g in range(len(self.weights[i][j])):
                    self.weights[i][j][g] = self.weights[i][j][g] - (self.lrnRate * self.errors[i][j] * self.struct[i][g])
                self.biases[i][j] = self.biases[i][j] - (self.lrnRate * self.errors[i][j])
        # print(self.weights)
        # print(self.biases)

    def StartLearning(self, epoch = 1):
        for i in range(epoch):
            print(i + 1)
            totalError = 0
            for j in range(len(self.inLayers)):
                self.struct[0] = self.inLayers[j]
                self.corrAnswers = self.answers[j]
                self.ForwardPropagation()
                self.OutputErrors()
                self.HiddenErrors()
                self.UpdateWeightsAndBiases()
                for k in range(len(self.struct[-1])):
                    totalError += (self.struct[-1][k] - self.corrAnswers[k]) ** 2
            self.epochErrors.append(totalError / len(self.inLayers))

    def predictQValues(self, state):
        self.struct[0] = state[:]
        self.ForwardPropagation()
        return self.struct[-1][:]

    def trainQLearning(self, state, action, reward, nextState, done, gamma=0.99):
        currentQValues = self.predictQValues(state)
        currentQ = currentQValues[action]
        nextQValues = self.predictQValues(nextState)
        maxNextQ = max(nextQValues)
        targetQ = reward
        if not done:
            targetQ += gamma * maxNextQ
        correctedQValues = currentQValues[:]
        correctedQValues[action] = targetQ
        self.struct[0] = state[:]
        self.corrAnswers = correctedQValues[:]
        self.ForwardPropagation()
        self.OutputErrors()
        self.HiddenErrors()
        self.UpdateWeightsAndBiases()


    def BildGrafic(self):
        import matplotlib.pyplot as plt
        plt.plot(range(len(self.epochErrors)), self.epochErrors)
        plt.xlabel('Эпоха')
        plt.ylabel('Средняя ошибка')
        plt.title('Обучение нейросети')
        plt.grid()
        plt.show()

    def Ask(self, inputLayer = [], convert = []):
        a = "y"
        while a == "y":
            if len(inputLayer) > 0:
                self.struct[0] = inputLayer
            else:
                b = input("Quetion: ").split()
                for i in range(len(b)):
                    self.struct[0][i] = float(b[i])
            self.ForwardPropagation()
            print(*self.struct[-1])
            for i in range(len(self.struct[-1])):
                if self.struct[-1][i] == max(self.struct[-1]):
                    if len(convert) > 0:
                        print(convert[i])
                    else:
                        print(i)
            a = input("Ask again?")
            while a != "n" and a != "y":
                a = input("Ask again?")

class cnn:
    class Conv2d:
        def __init__(self, numFilters = 1, filterSize = 3, inputDepth = 1, stride = 1, padding = 0, weights = [], biases = [], input = []):
            self.numFilters = numFilters
            self.filterSize = filterSize
            self.inputDepth = inputDepth
            self.stride = stride
            self.padding = padding
            self.weights = weights
            self.biases = biases
            self.input = input

            if len(self.weights) == 0:
                self.initialize()

        def padInput(self, input):
            if self.padding == 0:
                return  input

            out = []
            for c in range(self.inputDepth):
                channel = []
                for i in range(self.padding):
                    channel.append([0.0] * (len(input[0][0]) + self.padding * 2))
                for i in range(len(input[0])):
                    channel.append([0.0] * self.padding + input[c][i] + [0.0] * self.padding)
                for i in range(self.padding):
                    channel.append([0.0] * (len(input[0][0]) + self.padding * 2))
                out.append(channel)

            return  out

        def initialize(self):
            for f in range(self.numFilters):
                filter = []
                for c in range(self.inputDepth):
                    channel = []
                    for i in range(self.filterSize):
                        row = []
                        for j in range(self.filterSize):
                            row.append(random.uniform(-0.1, 0.1))
                        channel.append(row)
                    filter.append(channel)
                self.weights.append(filter)

            self.biases = [0.0] * self.numFilters

        def applyFilter(self, input, filter, bias):
            inH, inW = len(self.input[0]), len(self.input[0][0])
            self.outH = (inH - self.filterSize) // self.stride + 1
            self.outW = (inW - self.filterSize) // self.stride + 1
            out = []

            for i in range(self.outH):
                out.append([0.0] * self.outW)

            for i in range(self.outH):
                for j in range(self.outW):
                    value = 0.0
                    for c in range(self.inputDepth):
                        for y in range(self.filterSize):
                            for x in range(self.filterSize):
                                value += filter[c][y][x] * input[c][i * self.stride + y][j * self.stride + x]
                    out[i][j] = value + bias

            return out


        def forward(self, input):
            padded = self.padInput(input)
            self.input = padded
            out = []

            for f in range(self.numFilters):
                out.append(self.applyFilter(padded, self.weights[f], self.biases[f]))

            return out

        def backward(self, dOutput):
            self.dWeights = []
            self.dBiases = [0.0] * self.numFilters
            dInput = []

            for f in range(self.numFilters):
                filter = []
                for c in range(self.inputDepth):
                    channel = []
                    for y in range(self.filterSize):
                        row = [0.0] * self.filterSize
                        channel.append(row)
                    filter.append(channel)
                self.dWeights.append(filter)

            inH, inW = len(self.input[0]), len(self.input[0][0])
            for c in range(self.inputDepth):
                channel = []
                for i in range(inH):
                    channel.append([0.0] * inW)
                dInput.append(channel)

            for f in range(self.numFilters):
                for c in range(self.inputDepth):
                    for y in range(self.filterSize):
                        for x in range(self.filterSize):
                            self.dWeights[f][c][y][x] = 0
                            for i in range(self.outH):
                                for j in range(self.outW):
                                    self.dWeights[f][c][y][x] += dOutput[f][i][j] * self.input[c][i * self.stride + y][j * self.stride + x]

            for f in range(self.numFilters):
                self.dBiases[f] = 0.0
                for i in range(self.outH):
                    for j in range(self.outW):
                        self.dBiases[f] += dOutput[f][i][j]

            for c in range(self.inputDepth):
                for i in range(inH):
                    for j in range(inW):
                        dInput[c][i][j] = 0

                        for f in range(self.numFilters):
                            for y in range(self.filterSize):
                                for x in range(self.filterSize):
                                    oy = (i - y) / self.stride
                                    ox = (j - x) / self.stride

                                    if oy.is_integer() and ox.is_integer():
                                        oy = int(oy)
                                        ox = int(ox)
                                        if 0 <= oy < self.outH and 0 <= ox < self.outW:
                                            dInput[c][i][j] += self.weights[f][c][y][x] * dOutput[f][oy][ox]
        def updateWeights(self, learningRate):
            for f in range(self.numFilters):
                for c in range(self.inputDepth):
                    for y in range(self.filterSize):
                        for x in range(self.filterSize):
                            self.weights[f][c][y][x] -= learningRate * self.dWeights[f][c][y][x]
                self.biases[f] -= learningRate * self.dBiases[f]

        def save(self):
            return {
                "type": "Conv2d",
                "numFilters": self.numFilters,
                "filterSize": self.filterSize,
                "inputDepth": self.inputDepth,
                "stride": self.stride,
                "padding": self.padding,
                "weights": self.weights,
                "biases": self.biases,
                "input" : self.input
            }

    class ReLU:
        def __init__(self, input=[]):
            self.input = input

        def forward(self, input):
            self.input = input

            if isinstance(input[0], (int, float)):
                return [max(0, x) for x in input]

            out = []
            for c in range(len(input)):
                channel = []
                for i in range(len(input[c])):
                    row = []
                    for j in range(len(input[c][i])):
                        row.append(max(input[c][i][j], 0))
                    channel.append(row)
                out.append(channel)

            return out

        def backward(self, dOutput):
            if isinstance(self.input[0], (int, float)):
                return [dout if x > 0 else 0 for x, dout in zip(self.input, dOutput)]

            out = []
            for c in range(len(self.input)):
                channel = []
                for i in range(len(self.input[0])):
                    row = []
                    for j in range(len(self.input[0][0])):
                        if self.input[c][i][j] > 0:
                            row.append(dOutput[c][i][j])
                        else:
                            row.append(0)
                    channel.append(row)
                out.append(channel)

            return out

        def save(self):
            return {
                "type": "ReLU"
            }

    class MaxPool2d:
        def __init__(self, size=2, stride=1, input = [], output = []):
            self.size = size
            self.stride = stride
            self.input = input
            self.output = output

        def forward(self, input):
            inH, inW = len(input[0]), len(input[0][0])
            outH = (inH - self.size) // self.stride + 1
            outW = (inW - self.size) // self.stride + 1
            self.input = input
            self.output = []
            out = []

            for c in range(len(input)):
                channel = []
                for i in range(outH):
                    channel.append([0.0] * outW)
                for i in range(outH):
                    for j in range(outW):
                        values = []
                        for y in range(self.size):
                            for x in range(self.size):
                                values.append(input[c][i * self.stride + y][j * self.stride + x])
                        channel[i][j] = max(values)
                out.append(channel)
            self.output = out

            return out

        def backward(self, dOutput):
            out = []
            for c in range(len(self.input)):
                channel = []
                for i in range(len(self.input[0])):
                    channel.append([0.0] * len(self.input[0][0]))
                for i in range(len(dOutput[0])):
                    for j in range(len(dOutput[0][0])):
                        stop = False
                        for y in range(self.size):
                            if stop == True:
                                break
                            for x in range(self.size):
                                if self.input[c][i * self.stride + y][j * self.stride + x] == self.output[c][i][j]:
                                    channel[i * self.stride + y][j * self.stride + x] = dOutput[c][i][j]
                                    stop = True
                                    break
                out.append(channel)

            return out

        def save(self):
            return {
                "type": "MaxPool2d",
                "size": self.size,
                "stride": self.stride,
                "input" : self.input,
                "output" : self.output
            }

    class Flatten:
        def __init__(self, input = []):
            self.input = input

        def forward(self, input):
            self.input = input
            out = []
            for c in range(len(input)):
                for i in range(len(input[0])):
                    for j in range(len(input[0][0])):
                        out.append(input[c][i][j])

            return out

        def backward(self, dOutput):
            out = []
            index = 0
            channels, height, width = len(self.input), len(self.input[0]), len(self.input[0][0])

            for c in range(channels):
                channel = []
                for i in range(height):
                    row = []
                    for j in range(width):
                        row.append(dOutput[index])
                        index += 1
                    channel.append(row)
                out.append(channel)

            return  out

        def save(self):
            return {
                "type": "Flatten",
                "input" : self.input
            }

    class Linear:
        def __init__(self, inputSize, outputSize, weights=[], biases=[], input=[]):
            self.inputSize = inputSize
            self.outputSize = outputSize
            self.weights = weights
            self.biases = biases
            self.input = input

            if len(self.weights) == 0:
                self.initialize()

        def initialize(self):
            self.weights = []
            self.biases = []

            for i in range(self.outputSize):
                row = []
                for j in range(self.inputSize):
                    row.append(random.uniform(-0.1, 0.1))
                self.weights.append(row)

            self.biases = [0.0] * self.outputSize

        def forward(self, input):
            self.input = input
            out = []
            for i in range(self.outputSize):
                value = 0
                for j in range(self.inputSize):
                    value += self.weights[i][j] * input[j]
                out.append(value + self.biases[i])

            return out

        def backward(self, dOutput):
            self.dWeights = []
            self.dBiases = [0.0] * self.outputSize
            dInput = [0.0] * self.inputSize

            for i in range(self.outputSize):
                self.dWeights.append([0.0] * self.inputSize)

            for i in range(self.outputSize):
                for j in range(self.inputSize):
                    self.dWeights[i][j] = self.input[j] * dOutput[i]

            for i in range(len(dOutput)):
                self.dBiases[i] = dOutput[i]

            for j in range(self.inputSize):
                value = 0.0
                for i in range(self.outputSize):
                    value += self.weights[i][j] * dOutput[i]
                dInput[j] = value

            return dInput

        def updateWeights(self, learningRate):
            for i in range(len(self.weights)):
                for j in range(len(self.weights[0])):
                    self.weights[i][j] = self.weights[i][j] - learningRate * self.dWeights[i][j]

                self.biases[i] = self.biases[i] - learningRate * self.dBiases[i]

        def save(self):
            return {
                "type": "Linear",
                "inputSize": self.inputSize,
                "outputSize": self.outputSize,
                "weights": self.weights,
                "biases": self.biases
            }

    class Model:
        def __init__(self):
            self.layers = []

        def add(self, layer):
            self.layers.append(layer)

        def forward(self, input):
            for layer in self.layers:
                input = layer.forward(input)
            return input

        def backward(self, dOutput):
            for layer in reversed(self.layers):
                dOutput = layer.backward(dOutput)
            return dOutput

        def updateWeights(self, learningRate):
            for layer in self.layers:
                if hasattr(layer, 'updateWeights'):
                    layer.updateWeights(learningRate)

        def train(self, inputs, targets, learningRate, lossFn, lossFnDerivative):
            predictions = self.forward(inputs)
            loss = lossFn(predictions, targets)
            dLoss = lossFnDerivative(predictions, targets)
            self.backward(dLoss)
            self.updateWeights(learningRate)
            return loss

        def test(self, inputs):
            return self.forward(inputs)

    def softmax(logits):
        maxLogit = max(logits)
        exps = [math.exp(x - maxLogit) for x in logits]
        sum_exps = sum(exps)
        return [e / sum_exps for e in exps]

    def crossEntropyLoss(probs, target_one_hot):
        loss = 0.0
        for p, t in zip(probs, target_one_hot):
            if p > 0:
                loss -= t * math.log(p)
        return loss

    def crossEntropyDerivative(probs, target_one_hot):
        return [p - t for p, t in zip(probs, target_one_hot)]

    def yoloLoss(pred, target):
        return sum((p - t) ** 2 for p, t in zip(pred, target))

    def yoloLossDerivative(pred, target):
        return [2 * (p - t) for p, t in zip(pred, target)]

    def binaryCrossEntropy(predictions, targets):
        loss = 0.0
        for p, t in zip(predictions, targets):
            p = max(min(p, 1 - 1e-8), 1e-8)
            loss -= t * math.log(p) + (1 - t) * math.log(1 - p)
        return loss

    def binaryCrossEntropyDerivative(predictions, targets):
        derivatives = []
        for p, t in zip(predictions, targets):
            p = max(min(p, 1 - 1e-8), 1e-8)
            derivative = (p - t) / (p * (1 - p))
            derivatives.append(derivative)
        return derivatives

    class Data:
        def save(self, model, path = "", fileName = "cnnData"):
            import os
            import json
            os.makedirs(path, exist_ok=True)

            data = []
            for layer in model.layers:
                data.append(layer.save())
            with open(os.path.join(path, fileName + ".json"), "w") as f:
                json.dump(data, f, indent=4)

        def load(self, path):
            import json

            with open(path, "r") as f:
                data = json.load(f)

            model = cnn.Model()
            for layerData in data:
                if layerData["type"] == "Conv2d":
                    layer = cnn.Conv2d(numFilters=layerData["numFilters"], filterSize=layerData["filterSize"],
                                       inputDepth=layerData["inputDepth"], stride=layerData["stride"],
                                       padding=layerData["padding"], weights=layerData["weights"],
                                       biases=layerData["biases"], input=layerData["input"])
                elif layerData["type"] == "ReLU":
                    layer = cnn.ReLU(input=layerData["input"])
                elif layerData["type"] == "MaxPool2d":
                    layer = cnn.MaxPool2d(size=layerData["size"], stride=layerData["stride"], input=layerData["input"],
                                          output=layerData["output"])
                elif layerData["type"] == "Flatten":
                    layer = cnn.Flatten(input=layerData["input"])
                elif layerData["type"] == "Linear":
                    layer = cnn.Linear(inputSize=layerData["inputSize"], outputSize=layerData["outputSize"],
                                       weights=layerData["weights"], biases=layerData["biases"],
                                       input=layerData["input"])
                model.add(layer)

            return model

