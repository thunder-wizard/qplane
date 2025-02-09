import numpy as np
import random
import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

class QLearn():

    def __init__(self, n_stat, n_acts, gamm, lr, eps, dec, min, epsDecay, expName, saveForAutoReload, loadModel, usePredefinedSeeds, inputs, minReplay, replay, batch, update, stateDepth):
        self.n_states = n_stat
        self.n_actions = n_acts
        self.gamma = gamm
        self.learningRate = lr
        self.epsilon = eps
        self.decay = dec
        self.epsMin = min
        self.qTable = np.zeros([self.n_states, self.n_actions])
        self.n_epochsBeforeDecay = epsDecay
        self.experimentName = expName
        self.saveForAutoReload = saveForAutoReload

        if usePredefinedSeeds:
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)

        # Set device to GPU if available, otherwise CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model = DQNAgent(inputs, self.n_actions, self.learningRate,
                              minReplay, replay, batch, self.gamma, update, loadModel, stateDepth, self.device)

        self.modelSummary = self.model.modelSummary

        self.stateDepth = stateDepth
        self.id = "doubleDeep"
        self.currentTable = []

        self.numGPUs = torch.cuda.device_count()
        print("Num GPUs Available: ", self.numGPUs)

    def selectAction(self, state, episode):
        explorationTreshold = random.uniform(0, 1)
        explore = False
        if explorationTreshold > self.epsilon and len(state) == self.stateDepth:
            Qs = self.model.getQs(state)
            action = np.argmax(Qs)
        else:
            action = int(random.uniform(0, self.n_actions))
            explore = True
            Qs = ["Random"]

        self.currentTable = Qs

        if episode >= self.n_epochsBeforeDecay:
            if self.epsilon > self.epsMin:
                self.epsilon = self.epsilon * (1 - self.decay)
            elif self.epsilon < self.epsMin:
                self.epsilon = self.epsMin

        return action, explore, self.epsilon

    def learn(self, state, action, reward, new_state, done):
        if len(state) == self.stateDepth:
            self.model.updateReplayMemory((state, action, reward, new_state, done))
            self.model.train(done)

    def archive(self, epoch):
        if not os.path.exists("./Experiments/" + self.experimentName):
            os.makedirs("./Experiments/" + self.experimentName)
        torch.save(self.model.targetModel.state_dict(), "./Experiments/" + str(self.experimentName) + "/model" + str(epoch) + ".pth")
        replayMemFile = open("./Experiments/" + str(self.experimentName) + "/memory" + str(epoch) + ".pickle", 'wb')
        pickle.dump(self.model.replayMemory, replayMemFile)
        replayMemFile.close()
        if self.saveForAutoReload:
            torch.save(self.model.targetModel.state_dict(), "model.pth")
            replayMemFile = open("memory.pickle", 'wb')
            pickle.dump(self.model.replayMemory, replayMemFile)
            replayMemFile.close()


class DQNAgent:
    def __init__(self, inputs, outputs, learningRate, minReplay, replay, batch, gamma, update, loadModel, stateDepth, device):
        self.numOfInputs = inputs
        self.numOfOutputs = outputs
        self.learningRate = learningRate
        self.minReplayMemSize = minReplay
        self.replayMemSize = replay
        self.batchSize = batch
        self.gamma = gamma
        self.updateRate = update
        self.loadModel = loadModel
        self.loadModel = loadModel
        self.stateDepth = stateDepth
        self.device = device
        self.modelSummary = ""

        self.model = self.createModel().to(self.device)
        self.targetModel = self.createModel().to(self.device)
        self.targetModel.load_state_dict(self.model.state_dict())

        if self.loadModel:
            self.model.load_state_dict(torch.load(f"model.pth"))
            print("\nModel Loaded!\n")

        self.replayMemory = deque(maxlen=self.replayMemSize)

        if self.loadModel:
            replayMemFile = open(f"memory.pickle", 'rb')
            self.replayMemory = pickle.load(replayMemFile)
            replayMemFile.close()
            print("\nMemory Loaded!\n")

        self.targetUpdateCounter = 0

    def createModel(self):
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.numOfInputs * self.stateDepth, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.numOfOutputs)
        )
        return model

    def updateReplayMemory(self, transition):
        self.replayMemory.append(transition)

    def train(self, done):
        # Start training only if certain number of samples are already saved
        if len(self.replayMemory) < self.minReplayMemSize:
            return

        # Get a miniBatch of random samples from memory replay table
        miniBatch = random.sample(self.replayMemory, self.batchSize - 1)  # adds all but one samples at random to the minibatch
        miniBatch.append(self.replayMemory[-1])  # Adds the newest step to the minibatch

        # Get current states from miniBatch, then query NN model for Q values
        currentStates = torch.tensor([transition[0] for transition in miniBatch], dtype=torch.float32).to(self.device)
        currentQsList = self.model(currentStates)

        # Get future states from miniBatch, then query NN model for Q values
        newCurrentStates = torch.tensor([transition[3] for transition in miniBatch], dtype=torch.float32).to(self.device)
        futureQsListTarget = self.targetModel(newCurrentStates)  # Used for DDQN
        futureQsList = self.model(newCurrentStates)  # Used for DDQN

        statesInput = []
        controlsOutput = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(miniBatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            if not done:
                maxFutureQ = futureQsListTarget[index, torch.argmax(futureQsList[index]).item()]  # Used for DDQN
                new_q = reward + self.gamma * maxFutureQ
            else:
                new_q = reward

            # Update Q value for the given state
            currentQs = currentQsList[index].detach().clone()
            currentQs[action] = new_q

            # And append to our training data
            statesInput.append(current_state)
            controlsOutput.append(currentQs)

        # Convert to tensors and perform the batch update
        statesInput = torch.tensor(statesInput, dtype=torch.float32).to(self.device)
        controlsOutput = torch.stack(controlsOutput).to(self.device)

        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learningRate)

        # Perform training step
        optimizer.zero_grad()
        outputs = self.model(statesInput)
        loss = loss_fn(outputs, controlsOutput)
        loss.backward()
        optimizer.step()

        # Update target network counter every episode
        if done:
            self.targetUpdateCounter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.targetUpdateCounter > self.updateRate:
            self.targetModel.load_state_dict(self.model.state_dict())
            self.targetUpdateCounter = 0

    # Queries main network for Q values given current observation space (environment state)
    def getQs(self, state):
        if len(state) == self.stateDepth:
            state = torch.tensor([state], dtype=torch.float32).to(self.device)
            return self.model(state).detach().cpu().numpy()  # Move back to CPU for numpy compatibility
        else:
            return "State not Deep Enough yet"
