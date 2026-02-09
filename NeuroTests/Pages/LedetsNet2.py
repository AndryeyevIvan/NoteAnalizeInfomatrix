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
            self.weights = []
            self.biases = []
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

                                    if (i - y) % self.stride == 0 and (j - x) % self.stride == 0:
                                        oy = (i - y) // self.stride
                                        ox = (j - x) // self.stride
                                        if 0 <= oy < self.outH and 0 <= ox < self.outW:
                                            dInput[c][i][j] += self.weights[f][c][y][x] * dOutput[f][oy][ox]
            return dInput

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
                "biases": self.biases
            }

    class ReLU:
        def __init__(self, input = []):
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

    class LeakyReLU:
        def __init__(self, input = [], alpha = 0.1):
            self.input = input
            self.alpha = alpha

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
                        if input[c][i][j] > 0:
                            row.append(input[c][i][j])
                        else:
                            row.append(input[c][i][j] * self.alpha)
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
                            row.append(dOutput[c][i][j] * self.alpha)
                    channel.append(row)
                out.append(channel)

            return out

        def save(self):
            return {
                "type": "LeakyReLU",
                "alpha" : self.alpha
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
            inH, inW = len(self.input[0]), len(self.input[0][0])
            outH, outW = len(dOutput[0]), len(dOutput[0][0])

            for c in range(len(self.input)):
                channel = []
                for i in range(inH):
                    channel.append([0.0] * inW)
                for i in range(outH):
                    for j in range(outW):
                        stop = False
                        for y in range(self.size):
                            if stop:
                                break
                            for x in range(self.size):
                                inY = i * self.stride + y
                                inX = j * self.stride + x
                                if inY < inH and inX < inW:
                                    if self.input[c][inY][inX] == self.output[c][i][j]:
                                        channel[inY][inX] = dOutput[c][i][j]
                                        stop = True
                                        break
                out.append(channel)

            return out

        def save(self):
            return {
                "type": "MaxPool2d",
                "size": self.size,
                "stride": self.stride,
                "output" : self.output
            }

    class Upsample:
        def __init__(self, scale = 2, input = []):
            self.scale = scale
            self.input = input

        def forward(self, input):
            self.input = input
            out = []

            for c in range(len(input)):
                channel = []
                for i in range(len(input[0])):
                    row = []
                    for j in range(len(input[0][0])):
                        for s in range(self.scale):
                            row.append(input[c][i][j])
                    for s in range(self.scale):
                        channel.append(row[:])
                out.append(channel)

            return out

        def backward(self, dOutput):
            h, w = len(dOutput[0]) // self.scale, len(dOutput[0][0]) // self.scale
            out = []

            for c in range(len(dOutput)):
                channel = []
                for i in range(h):
                    channel.append([0] * w)
                out.append(channel)

            for c in range(len(dOutput)):
                for i in range(h):
                    for j in range(w):
                        value = 0
                        for y in range(self.scale):
                            for x in range(self.scale):
                                value += dOutput[c][i * self.scale + y][j * self.scale + x]
                        out[c][i][j] = value

            return out

        def save(self):
            return {
                "type": "Unsample",
                "scale" : self.scale
            }

    class Concat:
        def __init__(self, input = [], inputShapes = []):
            self.input = input
            self.inputShapes = inputShapes

        def forward(self, input):
            self.input = input
            self.inputShapes = [len(input[0]), len(input[1])]
            out = input[0] + input[1]

            return out

        def backward(self, dOutput):
            c1, c2 = self.inputShapes
            dOut1 = dOutput[:c1]
            dOut2 = dOutput[c1:]

            return [dOut1, dOut2]

        def save(self):
            return {
                "type": "Concat",
                "inputShapes": self.inputShapes
            }

    class Sigmoid:
        def __init__(self, input = []):
            self.input = input

        def sigmoid(self, x):
            return float(1 / (1 + math.exp(-x)))

        def forward(self, input = []):
            self.input = input
            if isinstance(input[0], (int, float)):
                return [self.sigmoid(x) for x in input]
            out = []

            for c in range(len(input)):
                channel = []
                for i in range(len(input[0])):
                    row = []
                    for j in range(len(input[0][0])):
                        row.append(self.sigmoid(input[c][i][j]))
                    channel.append(row)
                out.append(channel)

            return out

        def backward(self, dOutput):
            if isinstance(dOutput[0], (int, float)):
                out = []
                for i in range(len(self.input)):
                    sig = self.sigmoid(self.input[i])
                    out.append(sig * (1 - sig) * dOutput[i])

                return out

            out = []

            for c in range(len(dOutput)):
                channel = []
                for i in range(len(dOutput[0])):
                    row = []
                    for j in range(len(dOutput[0][0])):
                        sig = self.sigmoid(self.input[c][i][j])
                        grad = sig * (1 - sig) * dOutput[c][i][j]
                        row.append(grad)
                    channel.append(row)
                out.append(channel)

            return out

        def save(self):
            return {
                "type": "Sigmoid"
            }

    class Reshape:
        def __init__(self, targetShape = []):
            self.targetShape = targetShape
            self.inputShape = []

        def forward(self, input):
            self.inputShape = [len(input)]

            index = 0
            out = []

            for i in range(self.targetShape[0]):
                row = []
                for j in range(self.targetShape[1]):
                    row.append(input[index])
                    index += 1
                out.append(row)

            return out

        def backward(self, dOutput):
            flat = []
            for i in range(len(dOutput)):
                for j in range(len(dOutput[0])):
                    flat.append(dOutput[i][j])

            return flat

    class BatchNorm2d:
        def __init__(self, numChannels, epsilon=1e-5):
            self.numChannels = numChannels
            self.epsilon = epsilon
            self.gamma = [1.0] * numChannels
            self.beta = [0.0] * numChannels
            self.running_mean = [0.0] * numChannels
            self.running_var = [1.0] * numChannels

        def forward(self, input):
            self.input = input
            self.mean = []
            self.var = []
            self.normalized = []

            channels = len(input)
            height = len(input[0])
            width = len(input[0][0])

            for c in range(channels):
                m = sum([input[c][i][j] for i in range(height) for j in range(width)]) / (height * width)
                self.mean.append(m)

                v = sum([(input[c][i][j] - m) ** 2 for i in range(height) for j in range(width)]) / (height * width)
                self.var.append(v)

                channelNorm = []
                for i in range(height):
                    row = []
                    for j in range(width):
                        norm = (input[c][i][j] - m) / math.sqrt(v + self.epsilon)
                        row.append(norm)
                    channelNorm.append(row)
                self.normalized.append(channelNorm)

            output = []
            for c in range(channels):
                channel_out = []
                for i in range(height):
                    row = []
                    for j in range(width):
                        y = self.gamma[c] * self.normalized[c][i][j] + self.beta[c]
                        row.append(y)
                    channel_out.append(row)
                output.append(channel_out)

            return output

        def backward(self, dOutput):
            C = len(dOutput)
            H = len(dOutput[0])
            W = len(dOutput[0][0])
            N = H * W

            self.dGamma = [0.0] * C
            self.dBeta = [0.0] * C
            dInput = []

            for c in range(C):
                channel = []
                for i in range(H):
                    channel.append([0.0] * W)
                dInput.append(channel)

            for c in range(C):
                mean = self.mean[c]
                var = self.var[c]
                gamma = self.gamma[c]

                dBeta_c = 0.0
                dGamma_c = 0.0
                for i in range(H):
                    for j in range(W):
                        dout = dOutput[c][i][j]
                        norm = self.normalized[c][i][j]
                        dBeta_c += dout
                        dGamma_c += dout * norm
                self.dBeta[c] = dBeta_c
                self.dGamma[c] = dGamma_c

                dNorm = []
                for i in range(H):
                    row = []
                    for j in range(W):
                        row.append(dOutput[c][i][j] * gamma)
                    dNorm.append(row)

                sum_dNorm = 0.0
                sum_dNorm_xmu = 0.0
                for i in range(H):
                    for j in range(W):
                        x_mu = self.input[c][i][j] - mean
                        sum_dNorm += dNorm[i][j]
                        sum_dNorm_xmu += dNorm[i][j] * x_mu

                std_inv = 1.0 / math.sqrt(var + self.epsilon)

                for i in range(H):
                    for j in range(W):
                        x_mu = self.input[c][i][j] - mean
                        term1 = dNorm[i][j] * std_inv
                        term2 = (1.0 / N) * sum_dNorm_xmu * x_mu * (std_inv ** 3)
                        term3 = (1.0 / N) * sum_dNorm
                        dInput[c][i][j] = term1 - term2 - term3

            return dInput

        def updateWeights(self, learningRate):
            for c in range(self.numChannels):
                self.gamma[c] -= learningRate * self.dGamma[c]
                self.beta[c] -= learningRate * self.dBeta[c]

        def save(self):
            return {
                "type": "BatchNorm2d",
                "numChannels": self.numChannels,
                "gamma": self.gamma,
                "beta": self.beta
            }

    class BatchNorm1d:
        def __init__(self, numFeatures, epsilon=1e-5):
            self.numFeatures = numFeatures
            self.epsilon = epsilon
            self.gamma = [1.0] * numFeatures  # параметр масштаба
            self.beta = [0.0] * numFeatures  # параметр сдвига
            # Для инференса можно добавить running_mean и running_var

        def forward(self, input):
            """
            input: одномерный список [f1, f2, ..., fn] где n = numFeatures
            """
            self.input = input
            N = len(input)

            # Вычисляем mean и var по всем элементам вектора
            self.mean = sum(input) / N
            self.var = sum((x - self.mean) ** 2 for x in input) / N

            # Нормализуем
            self.normalized = []
            for x in input:
                norm = (x - self.mean) / math.sqrt(self.var + self.epsilon)
                self.normalized.append(norm)

            # Масштабируем и сдвигаем (learnable parameters)
            output = []
            for i, norm_val in enumerate(self.normalized):
                y = self.gamma[i] * norm_val + self.beta[i]
                output.append(y)

            return output

        def backward(self, dOutput):
            N = len(self.input)

            self.dGamma = [0.0] * self.numFeatures
            self.dBeta = [0.0] * self.numFeatures

            for i in range(N):
                self.dGamma[i] += dOutput[i] * self.normalized[i]
                self.dBeta[i] += dOutput[i]

            dInput = [0.0] * N

            sum_dNorm = 0.0
            sum_dNorm_xmu = 0.0
            for i in range(N):
                dNorm_val = dOutput[i] * self.gamma[i]
                x_mu = self.input[i] - self.mean
                sum_dNorm += dNorm_val
                sum_dNorm_xmu += dNorm_val * x_mu

            std_inv = 1.0 / math.sqrt(self.var + self.epsilon)

            for i in range(N):
                x_mu = self.input[i] - self.mean
                dNorm_val = dOutput[i] * self.gamma[i]

                term1 = dNorm_val * std_inv
                term2 = (1.0 / N) * sum_dNorm_xmu * x_mu * (std_inv ** 3)
                term3 = (1.0 / N) * sum_dNorm * std_inv

                dInput[i] = term1 - term2 - term3

            return dInput

        def updateWeights(self, learningRate):
            for i in range(self.numFeatures):
                self.gamma[i] -= learningRate * self.dGamma[i]
                self.beta[i] -= learningRate * self.dBeta[i]

        def save(self):
            return {
                "type": "BatchNorm1d",
                "numFeatures": self.numFeatures,
                "gamma": self.gamma,
                "beta": self.beta
            }

    class Residual:
        def __init__(self, filters):
            self.conv1 = cnn.Conv2d(filters, 1, stride=1)
            self.conv2 = cnn.Conv2d(filters * 2, 3, stride=1)
            self.bn1 = cnn.BatchNorm2d(filters)
            self.bn2 = cnn.BatchNorm2d(filters * 2)
            self.leaky = cnn.LeakyReLU(alpha=0.1)
            self.residual = None

        def forward(self, x):
            self.residual = x
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.leaky(out)
            out = self.conv2(out)
            out = self.bn2(out)
            out += self.residual
            return self.leaky(out)

        def backward(self, dOutput):
            dOutput = self.leaky.backward(dOutput)
            dResidual = dOutput


            dOut = self.bn2.backward(dOutput)
            dOut = self.conv2.backward(dOut)
            dOut = self.leaky.backward(dOut)
            dOut = self.bn1.backward(dOut)
            dOut = self.conv1.backward(dOut)

            return dOut + dResidual

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
                "type": "Flatten"
            }

    class Linear:
        def __init__(self, inputSize, outputSize , weights = [], biases = [], input = []):
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

    class GlobalAveragePooling:
        def __init__(self):
            self.input = None

        def forward(self, x):
            self.input = x
            output = []
            for channel in x:  # Для каждого канала
                total = 0
                count = 0
                for row in channel:
                    for pixel in row:
                        total += pixel
                        count += 1
                output.append(total / count)  # Усредняем весь канал
            return output

        def backward(self, dOutput):
            # dOutput: [channels] - градиенты от следующего слоя
            channels, height, width = len(self.input), len(self.input[0]), len(self.input[0][0])
            dInput = []

            for c in range(channels):
                channel_grad = []
                for i in range(height):
                    row = []
                    for j in range(width):
                        # Распределяем градиент равномерно по всем пикселям канала
                        row.append(dOutput[c] / (height * width))
                    channel_grad.append(row)
                dInput.append(channel_grad)

            return dInput

        def save(self):
            return {
                "type": "GlobalAveragePooling"
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

    def binaryCrossEntropy(pred, target):
        loss = 0
        for p, t in zip(pred, target):
            p = max(min(p, 1 - 1e-9), 1e-9)
            loss += -(t * math.log(p) + (1 - t) * math.log(1 - p))
        return loss / len(pred)

    def binaryCrossEntropyDerivative(pred, target):
        n = len(pred)
        return [(p - t) / ((p * (1 - p) + 1e-9) * n) for p, t in zip(pred, target)]

    def focalLoss(pred, target, alpha=0.25, gamma=2):
        eps = 1e-9
        loss = 0.0

        for p, t in zip(pred, target):
            p = max(min(p, 1 - eps), eps)

            if t == 1:
                loss += -alpha * ((1 - p) ** gamma) * math.log(p)
            else:
                loss += -(1 - alpha) * (p ** gamma) * math.log(1 - p)

        return loss / len(pred)

    def focalLossDerivative(pred, target, alpha=0.25, gamma=2):
        eps = 1e-9
        grads = []
        n = len(pred)

        for p, t in zip(pred, target):
            p = max(min(p, 1 - eps), eps)

            if t == 1:
                grad = -alpha * (gamma * ((1 - p) ** (gamma - 1)) * (-1) * math.log(p) + ((1 - p) ** gamma) * (1 / p))
            else:
                grad = -(1 - alpha) * (gamma * (p ** (gamma - 1)) * math.log(1 - p) + (p ** gamma) * (-1 / (1 - p)))

            grads.append(grad / n)

        return grads

    class Data:
        def save(self, model, path="", fileName="cnnData"):
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
                                       biases=layerData["biases"])
                elif layerData["type"] == "ReLU":
                    layer = cnn.ReLU()
                elif layerData["type"] == "LeakyReLU":
                    layer = cnn.LeakyReLU(alpha=layerData.get("alpha", 0.1))
                elif layerData["type"] == "MaxPool2d":
                    layer = cnn.MaxPool2d(size=layerData["size"], stride=layerData["stride"])
                elif layerData["type"] == "Flatten":
                    layer = cnn.Flatten()
                elif layerData["type"] == "Linear":
                    layer = cnn.Linear(inputSize=layerData["inputSize"], outputSize=layerData["outputSize"],
                                       weights=layerData["weights"], biases=layerData["biases"])
                elif layerData["type"] == "BatchNorm2d":
                    layer = cnn.BatchNorm2d(numChannels=layerData["numChannels"],
                                            epsilon=layerData.get("epsilon", 1e-5))
                    layer.gamma = layerData["gamma"]
                    layer.beta = layerData["beta"]
                elif layerData["type"] == "BatchNorm1d":
                    layer = cnn.BatchNorm1d(numFeatures=layerData["numFeatures"],
                                            epsilon=layerData.get("epsilon", 1e-5))
                    layer.gamma = layerData["gamma"]
                    layer.beta = layerData["beta"]
                elif layerData["type"] == "GlobalAveragePooling":
                    layer = cnn.GlobalAveragePooling()
                elif layerData["type"] == "Sigmoid":
                    layer = cnn.Sigmoid()
                model.add(layer)

            return model