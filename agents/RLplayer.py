import time

from numpy import sign

from template import Agent
import heapq
import math
from Sequence.sequence_model import *
import json
"""
Authors: Group-28 Unimelb comp90054 2021s1

"""
class NoneDict(dict):
    def __getitem__(self, idx):
        self.setdefault(idx, 0.0)
        return dict.__getitem__(self, idx)
    # def __getitem__(self, key):
    #     if dict.get(self, key) is None:
    #         return 0


class myAgent(Agent):
    def __init__(self, _id):
        super().__init__(_id)
        #: TODO: not sure how to use
        self.episodesSoFar = 0
        self.accumTrainRewards = 0.0
        self.accumTestRewards = 0.0
        # set some essential arguments
        # epsilon: exploration rate, gamma: future discount
        # alpha: learning rate,
        # numEpisode: number of training episodes
        # qValue: to store the Q-value of (s,a)
        self.epsilon = 0.05
        self.gamma = 0.9
        self.alpha = 0.1
        self.numEpisode = 200
        self.qValues = NoneDict()
    #
    # def test(self):
    #     print("test")

    def SelectAction(self, actions, game_state):

        whole_state = (game_state, actions)
        action = random.choice(actions)
        # TODO: future desgin a decreasing epsilon
        if random.random() > self.epsilon:
            action = self.getPolicy(whole_state)
        self.doAction(whole_state, action)
        #print(self.qValues)
        return action

    def getPolicy(self,whole_state):
        # to get the best list of actions
        maxValue = -math.inf
        actions = whole_state[1]
        argMaxAction = random.choice(actions)
        if len(actions) == 0:
            # TODO: should it return none or {'play_card':None...}
            return None
        for action in actions:
            actionString = json.dumps(action)
            if maxValue < self.getQValue(whole_state[0], actionString):
                maxValue = self.getQValue(whole_state[0], actionString)
                argMaxAction = action
        return argMaxAction

    def updateQValue(self, whole_state, next_state, action, reward):
        # TODO: reward shaping can be added in future
        actionString = json.dumps(action)
        print(actionString)
        self.qValues[(whole_state[0], actionString)] = self.getQValue(whole_state[0], actionString) + \
                self.alpha * (reward + self.gamma * self.getValue(next_state) -
                              self.getQValue(whole_state[0], actionString))

    def getQValue(self, game_state, actionString):
        return self.qValues[(game_state, actionString)]

    def getValue(self, whole_state):
        # V(s) = max(Q(s,a)) get the maximum Q(s,a) of current s
        maxValue = -math.inf
        actions = whole_state[1]
        for action in actions:
            actionString = json.dumps(action)
            maxValue = max(maxValue, self.getQValue(whole_state[0], actionString))

        return maxValue


###learningAgent

    def obeserveTransition(self, whole_state, action, next_state, deltaReward):
        self.epoch_reward += deltaReward
        self.updateQValue(whole_state, next_state, action,deltaReward)

    # TODO: start and end suppose to be present the average reward only
    def startEpoch(self):
        # starting a new round
        self.lastState = None
        self.lastAction = None
        self.epoch_reward = 0.0

    def endEpoch(self):
        if self.episodesSoFar < self.numEpisode:
            self.accumTrainRewards += self.epoch_reward
        else:
            self.accumTestRewards += self.epoch_reward
        self.episodesSoFar+=1
        if self.episodesSoFar >= self.numEpisode:
            # follow policy
            self.epsilon = 0.0
            self.alpha = 0.0

    # TODO: not sure where to use
    def isInTraining(self):
        return self.episodesSoFar < self.numEpisode

    def isInTesting(self):
        return not self.isInTraining()

    def doAction(self, whole_state, action):
        """
            Called by inherited class when
            an action is taken in a state
        """
        self.lastState = whole_state
        self.lastAction = action

    ## for sequence
    def observationFunction(self,whole_state):

        if not self.lastState is None:
            game_state = whole_state[0]
            last_game_state = self.lastState[0]
            reward = game_state.agents[self.id].score -\
                     last_game_state.agents[self.id].score
            self.obeserveTransition(self.lastState,self.lastAction,whole_state,reward)
        return whole_state

    def register(self,whole_state):
        self.startEpoch()
        if self.episodesSoFar == 0:
            print("Beginning %d episodes of Training"% (self.numEpisode))

    def final(self,whole_state):

        game_state = whole_state[0]
        last_game_state = self.lastState[0]
        deltaReward = game_state.agents[self.id].score -\
                     last_game_state.agents[self.id].score
        self.obeserveTransition(self.lastState, self.lastAction, whole_state, deltaReward)
        self.endEpoch()
        print(self.qValues)
        # # Make sure we have this var
        # if not 'episodeStartTime' in self.__dict__:
        #     self.episodeStartTime = time.time()
        # if not 'lastWindowAccumRewards' in self.__dict__:
        #     self.lastWindowAccumRewards = 0.0
        # self.lastWindowAccumRewards += game_state.agents[self.id].score
        #
        # NUM_EPS_UPDATE = 100
        # if self.episodesSoFar % NUM_EPS_UPDATE == 0:
        #     print('Reinforcement Learning Status:')
        #
        #     windowAvg = self.lastWindowAccumRewards / float(NUM_EPS_UPDATE)
        #     if self.episodesSoFar <= self.numTraining:
        #         trainAvg = self.accumTrainRewards / float(self.episodesSoFar)
        #         print('\tCompleted %d out of %d training episodes' % (
        #             self.episodesSoFar, self.numEpisode))
        #
        #         print('\tAverage Rewards over all training: %.2f' % (
        #             trainAvg))
        #     else:
        #         testAvg = float(self.accumTestRewards) / (self.episodesSoFar - self.numEpisode)
        #         print('\tCompleted %d test episodes' % (self.episodesSoFar - self.numEpisode))
        #         print('\tAverage Rewards over testing: %.2f' % testAvg)
        #
        #     print('\tAverage Rewards for last %d episodes: %.2f' % (
        #         NUM_EPS_UPDATE, windowAvg))
        #
        #     print('\tEpisode took %.2f seconds' % (time.time() - self.episodeStartTime))
        #
        #     self.lastWindowAccumRewards = 0.0
        #     self.episodeStartTime = time.time()


        if self.episodesSoFar == self.numEpisode:
            msg = 'Training Done (turning off epsilon and alpha)'
            print('%s\n%s' % (msg, '-' * len(msg)))