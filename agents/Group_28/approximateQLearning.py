from numpy import sign
from template import Agent
import heapq
from Sequence.sequence_model import *
import random

"""
Authors: Group-28 Unimelb comp90054 2021s1
If want to train this model, need to use game_test.py and sequence_runner_test.py, and uncomment the weight output file

"""
EPSILON = 0.05
GAMMA = 0.9
ALPHA = 0.005

class myAgent(Agent):
    def __init__(self, _id):
        super().__init__(_id)
        self.draft_weight = {"draft-take-two-eyed":200,"draft-take-one-eyed":50,"draft-seq-num":45,"draft-closest-friend-distance":5}
        self.remove_weight = {"remove-hearts":100,"remove-seq-num":60,"remove-closest-friend-distance":10,"remove-opp-num":2}
        self.play_weight = {"play-hearts":50,"play-seq-num":100,"play-closest-friend-distance":10,"play-opp-seq-num":1}
        print("initial success")
        # self.draft_weight = AdvancedDict()
        # self.remove_weight = AdvancedDict()
        # self.play_weight = AdvancedDict()

    def SelectAction(self, actions, game_state):

        whole_state = (game_state, actions)
        action = random.choice(actions)
        if random.random() > EPSILON:
            action = self.getPolicy(whole_state)
        self.doAction(whole_state, action)
        return action

    def getPolicy(self, whole_state):
        # to get the best list of actions
        print("in the getPolicy")
        actions = whole_state[1]
        argMaxAction = random.choice(actions)

        if len(actions) == 0:
            # TODO: should it return none or {'play_card':None...}
            return None

        maxValue = 0
        # try to find an action that contains the max q value of draft card
        for action in actions:
            print("finding best draft")
            if maxValue < self.getQValue("draft", self.draft_weight, whole_state, action):
                maxValue = self.getQValue("draft", self.draft_weight, whole_state, action)
                argMaxAction = action
        best_draft_card = argMaxAction["draft_card"]
        maxValue = 0
        # try to find all actions that contains this draft card
        for action in actions:
            if action["draft_card"] == best_draft_card:
                # if got trade then throw it immediately and choose the best draft card
                if action["type"] == "trade":
                    return action
                # calculate we should do remove or place
                if action["type"] == "remove":
                    remove_q = self.getQValue("remove", self.remove_weight, whole_state, action)
                    if maxValue < remove_q:
                        maxValue = remove_q
                        argMaxAction = action
                if action["type"] == "place":
                    play_q = self.getQValue("play", self.play_weight, whole_state, action)
                    if maxValue < play_q:
                        maxValue = play_q
                        argMaxAction = action
        return argMaxAction

    def getValue(self, feature_name, whole_state):
        """
        Get the value of current state, which is the maxQ(s,a)
        """
        maxValue = 0
        actions = whole_state[1]

        if feature_name == "draft":
            for action in actions:
                maxValue = max(maxValue, self.getQValue(feature_name, self.draft_weight, whole_state, action))
        elif feature_name == "remove":
            for action in actions:
                maxValue = max(maxValue, self.getQValue(feature_name, self.remove_weight, whole_state, action))
        else:
            for action in actions:
                maxValue = max(maxValue, self.getQValue(feature_name, self.play_weight, whole_state, action))

        return maxValue

    def observationFunction(self, whole_state):
        if not self.lastState is None:
            game_state = whole_state[0]
            last_game_state = self.lastState[0]
            # not only care about winning but also need to penalize if put
            # wrong place and let the next agent win
            reward = game_state.agents[self.id].score - \
                     last_game_state.agents[self.id].score
            penalize = game_state.agents[(self.id + 1) % 4].score - \
                       last_game_state.agents[(self.id + 1) % 4].score
            self.obeserveTransition(self.lastState, self.lastAction, whole_state, reward - penalize / 5)
        return whole_state

    def obeserveTransition(self, whole_state, action, next_state, deltaReward):
        # make reward become larger
        self.updateQValue(whole_state, next_state, action, deltaReward * 100)

    def updateQValue(self, whole_state, next_state, action, reward):
        # update three q values respectively
        draft_feature, remove_feature, play_feature = self.getFeatures(whole_state, action)
        for key in draft_feature.keys():
            self.draft_weight[key] += draft_feature[key] * ALPHA * \
                                      (reward + GAMMA * self.getValue("draft",next_state) -
                                       self.getQValue("draft", self.draft_weight,
                                                      whole_state, action))
        for key in remove_feature.keys():
            self.remove_weight[key] += remove_feature[key] * ALPHA * \
                                       (reward + GAMMA * self.getValue("remove",next_state) -
                                        self.getQValue("remove", self.remove_weight,
                                                       whole_state, action))
        for key in play_feature.keys():
            self.play_weight[key] += play_feature[key] * ALPHA * \
                                     (reward + GAMMA * self.getValue("play`",next_state) -
                                      self.getQValue("play", self.play_weight,
                                                     whole_state, action))

        print("draft weight:", self.draft_weight)
        print("remove weight:", self.remove_weight)
        print("play weight:", self.play_weight)
#         f = open("QlearnWeight.txt", 'w')
#         f.write("draft weight:")
#         f.write(str(self.draft_weight))
#         f.write("\n")
#         f.write("remove weight:")
#         f.write(str(self.remove_weight))
#         f.write("\n")
#         f.write("play weight:")
#         f.write(str(self.play_weight))
#         f.write("\n")

    def getQValue(self, feature_name, weights, whole_state, action):
        print("In get Qvalue and feature name is",feature_name)
        qValue = 0.0
        draft_feature, remove_feature, play_feature = self.getFeatures(whole_state, action)
        if feature_name == "draft":
            features = draft_feature
        elif feature_name == "remove":
            features = remove_feature
        else:
            features = play_feature

        for key in features.keys():
            qValue += (weights[key] * features[key])
        print("qValue", qValue)
        return qValue

    def getFeatures(self, whole_state, action):
        print("in get feature")
        draftcard = action["draft_card"]
        type = action["type"]
        chips = whole_state[0].board.chips
        position = action["coords"]

        remove_feature = AdvancedDict()
        play_feature = AdvancedDict()
        draft_feature = self.draftFeature(draftcard, chips, whole_state[0])
        if type == "remove":
            remove_feature = self.removeFeature(position, chips, whole_state[0])
        elif type == "place":
            play_feature = self.playFeature(position, chips, whole_state[0])

        return draft_feature, remove_feature, play_feature

    def draftFeature(self, card, chips, game_state):
        """
        Extract the feature of draft card, two-eyed, one-eyed, number of sequence, distance to closest friend
        """
        print("in draft feature")
        feature = AdvancedDict()
        plr_state = game_state.agents[self.id]
        print("plr_state is:",plr_state)
        color = plr_state.colour
        if card == 'jd' or card == 'jc':
            # if two-eyed in the draft, we take it immediately
            feature["draft-take-two-eyed"] = 1.0
        if card == 'js' or card == 'jh':
            # if one-eyed in the draft, second priority
            feature["draft-take-one-eyed"] = 1.0
        positions = COORDS[card]
        print("position is ",positions)
        for x, y in positions:
            if chips[x][y] == EMPTY:
                # pretend to play to see if can make a sequence
                chips[x][y] = color
                # {'num_seq':num_seq, 'orientation':[k for k,v in seq_found.items() if v], 'coords':seq_coords},
                # seq_type
                print("try to call checkSeq")
                seq_info = self.checkSeq(chips, plr_state, (x, y))
                print("calling checkSeq success and info is:",seq_info)
                chips[x][y] = EMPTY
                feature["draft-seq-num"] = float(max(seq_info["num_seq"], feature["draft-seq-num"]))
                feature["draft-closest-friend-distance"] = min(self.uscActionsA((x,y), game_state, True)*10,
                                                               feature["draft-closest-friend-distance"])
        return feature

    def removeFeature(self, position, chips, game_state):
        """
        Extract the feature of removing, removing opp chips on 4 hearts, remove opp that has many fellows around
        remove opp that can give a mini distance between friends
        """
        print("in remove feature")
        feature = AdvancedDict()
        color = game_state.agents[self.id].colour
        opp_color = game_state.agents[(self.id + 1) % 4].opp_colour
        plr_state = game_state.agents[self.id]
        awesomeList = [(4, 4), (4, 5), (5, 4), (5, 5)]

        # if we are removing the hearts, good move!!
        if position in awesomeList:
            feature["remove-hearts"] = 1.0
        # if not heart, we should remove the pos that if we put a chip there, we can
        # make a sequence
        x, y = position
        chips[x][y] = color
        seq_info = self.checkSeq(chips, plr_state, (x, y))
        chips[x][y] = EMPTY
        feature["remove-seq-num"] = float(max(seq_info["num_seq"], feature["remove-seq-num"]))
        feature["remove-closest-friend-distance"] = min(self.uscActionsA(position, game_state, True)*10,
                                                        feature["remove-closest-friend-distance"])
        feature["remove-opp-num"] = self.oppAlmostSeq(position, chips, opp_color)
        return feature

    def playFeature(self, position, chips, game_state):
        """
        Extract the play card features, choose 4 heart to place, play the card that can make a sequence immediately,
        play the card that close to its friend, play tha card that prevent opp to make a sequence
        NUM_PLAY_WEIGHT = 4
        """
        print("in play feature")
        feature = AdvancedDict()
        color = game_state.agents[self.id].colour
        opp_color = game_state.agents[(self.id + 1) % 4].opp_colour
        plr_state = game_state.agents[self.id]
        opp_plr_state = game_state.agents[(self.id + 1) % 4]
        awesomeList = [(4, 4), (4, 5), (5, 4), (5, 5)]

        # if we are placing on the hearts, good move!!
        if position in awesomeList:
            feature["play-hearts"] = 1.0
        # if not heart, we should place the pos that if we put a chip there, we can
        # make a sequence
        x, y = position
        chips[x][y] = color
        seq_info = self.checkSeq(chips, plr_state, (x, y))
        chips[x][y] = EMPTY
        feature["play-seq-num"] = float(max(seq_info["num_seq"], feature["play-seq-num"]))
        feature["play-closest-friend-distance"] = min(self.uscActionsA(position, game_state, True)*10,
                                                      feature["play-closest-friend-distance"])

        chips[x][y] = opp_color
        seq_info = self.checkSeq(chips, opp_plr_state, (x, y))
        chips[x][y] = EMPTY
        feature["play-opp-seq-num"] = float(max(seq_info["num_seq"], feature["play-opp-seq-num"]))
        return feature

    ###learningAgent

    # TODO: start and end suppose to be present the average reward only
    def startEpoch(self):
        # starting a new round
        self.lastState = None
        self.lastAction = None

    def doAction(self, whole_state, action):
        """
            Called by inherited class when
            an action is taken in a state
        """
        self.lastState = whole_state
        self.lastAction = action

    def register(self, whole_state):
        self.startEpoch()

    def final(self, whole_state):
        game_state = whole_state[0]
        last_game_state = self.lastState[0]
        deltaReward = game_state.agents[self.id].score - \
                      last_game_state.agents[self.id].score
        self.obeserveTransition(self.lastState, self.lastAction, whole_state, deltaReward)


    def checkSeq(self, chips, plr_state, last_coords):
        """
        Copy from sequence_model for check if there ganna be a sequence
        """
        clr, sclr = plr_state.colour, plr_state.seq_colour
        oc, os = plr_state.opp_colour, plr_state.opp_seq_colour
        seq_type = TRADSEQ
        seq_coords = []
        seq_found = {'vr': 0, 'hz': 0, 'd1': 0, 'd2': 0, 'hb': 0}
        found = False
        nine_chip = lambda x, clr: len(x) == 9 and len(set(x)) == 1 and clr in x
        lr, lc = last_coords

        # All joker spaces become player chips for the purposes of sequence checking.
        for r, c in COORDS['jk']:
            chips[r][c] = clr

        # First, check "heart of the board" (2h, 3h, 4h, 5h). If possessed by one team, the game is over.
        coord_list = [(4, 4), (4, 5), (5, 4), (5, 5)]
        heart_chips = [chips[y][x] for x, y in coord_list]
        if EMPTY not in heart_chips and (clr in heart_chips or sclr in heart_chips) and not (
                oc in heart_chips or os in heart_chips):
            seq_type = HOTBSEQ
            seq_found['hb'] += 2
            seq_coords.append(coord_list)

        # Search vertical, horizontal, and both diagonals.
        vr = [(-4, 0), (-3, 0), (-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]
        hz = [(0, -4), (0, -3), (0, -2), (0, -1), (0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]
        d1 = [(-4, -4), (-3, -3), (-2, -2), (-1, -1), (0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
        d2 = [(-4, 4), (-3, 3), (-2, 2), (-1, 1), (0, 0), (1, -1), (2, -2), (3, -3), (4, -4)]
        for seq, seq_name in [(vr, 'vr'), (hz, 'hz'), (d1, 'd1'), (d2, 'd2')]:
            coord_list = [(r + lr, c + lc) for r, c in seq]
            coord_list = [i for i in coord_list if 0 <= min(i) and 9 >= max(i)]  # Sequences must stay on the board.
            chip_str = ''.join([chips[r][c] for r, c in coord_list])
            # Check if there exists 4 player chips either side of new chip (counts as forming 2 sequences).
            if nine_chip(chip_str, clr):
                seq_found[seq_name] += 2
                seq_coords.append(coord_list)
            # If this potential sequence doesn't overlap an established sequence, do fast check.
            if sclr not in chip_str:
                sequence_len = 0
                start_idx = 0
                for i in range(len(chip_str)):
                    if chip_str[i] == clr:
                        sequence_len += 1
                    else:
                        start_idx = i + 1
                        sequence_len = 0
                    if sequence_len >= 5:
                        seq_found[seq_name] += 1
                        seq_coords.append(coord_list[start_idx:start_idx + 5])
                        break
            else:  # Check for sequences of 5 player chips, with a max. 1 chip from an existing sequence.
                for pattern in [clr * 5, clr * 4 + sclr, clr * 3 + sclr + clr, clr * 2 + sclr + clr * 2,
                                clr + sclr + clr * 3, sclr + clr * 4]:
                    for start_idx in range(5):
                        if chip_str[start_idx:start_idx + 5] == pattern:
                            seq_found[seq_name] += 1
                            seq_coords.append(coord_list[start_idx:start_idx + 5])
                            found = True
                            break
                    if found:
                        break

        for r, c in COORDS['jk']:
            chips[r][c] = JOKER  # Joker spaces reset after sequence checking.

        num_seq = sum(seq_found.values())
        if num_seq > 1 and seq_type != HOTBSEQ:
            seq_type = MULTSEQ
        return {'num_seq': num_seq, 'orientation': [k for k, v in seq_found.items() if v], 'coords': seq_coords}

    def oppAlmostSeq(self, position, chips, opp_color):
      """
      Calculate in four directions, how many opponents are there
      """
        seqs = [[(-4, 0), (-3, 0), (-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0), (3, 0), (4, 0)],
                [(0, -4), (0, -3), (0, -2), (0, -1), (0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],
                [(-4, -4), (-3, -3), (-2, -2), (-1, -1), (0, 0), (1, 1), (2, 2), (3, 3), (4, 4)],
                [(-4, 4), (-3, 3), (-2, 2), (-1, 1), (0, 0), (1, -1), (2, -2), (3, -3), (4, -4)]]
        x, y = position
        opp_num_max = 0
        for seq in seqs:
            opp_num = 0
            for i in range(len(seq)):
                dx, dy = seq[i]
                if x + dx >= 0 and x + dx < 10 and y + dy >= 0 and y + dy < 10:
                    if chips[x + dx][y + dy] == opp_color:
                        opp_num += 1
                    opp_num_max = max(opp_num_max, opp_num)
        return opp_num_max

    #  mini distance between one of the possible action and the goal
    def uscActionsA(self, point, game_state, isFriend):
        if point is None:
            return 3
        point1s = []
        minDistance = 9

        awesomeSet = set()
        awesomeSet.update(((4, 4), (4, 5), (5, 4), (5, 5), (0, 0), (0, 9), (9, 9), (9, 0)))

        # if the agent one before it has no last action, means it is the first one
        # then we need to find the closest distance between 4 hearts/4 corner with
        # all the possible actions
        if game_state.agents[(self.id - 1) % 4].last_action == None:
            if not isFriend:
                return 9
            point1s = list(awesomeSet)
            print("point1s:", point1s)
            minDistance = min(self.chipDistanceH(point, point1s, game_state), minDistance)
            print(point)
            print("first round distance:", minDistance)
            return minDistance
        else:
            color = game_state.agents[self.id].colour
            seqColor = game_state.agents[self.id].seq_colour
            opp_color = game_state.agents[self.id].opp_colour
            opp_seqColor = game_state.agents[self.id].opp_seq_colour
            chips = game_state.board.chips

            # get the current position of same color chips and JOKER and empty 4 Hearts
            for x in range(len(chips)):
                for y in range(len(chips[0])):
                    if isFriend:
                        if chips[x][y] == color or chips[x][y] == seqColor:
                            point1s.append((x, y))
                        if (x, y) in awesomeSet and chips[x][y] == EMPTY:
                            point1s.append((x, y))
                        # append four corner
                        point1s.append((x, y))
                    if not isFriend:
                        if chips[x][y] == opp_color or chips[x][y] == opp_seqColor:
                            point1s.append((x, y))

            # try to get closer with same color chips or JOKER or 4 Hearts
            # consider the obstacle(opp_color) in the way, and four directions
            minDistance = min(self.chipDistanceH(point, point1s, game_state), minDistance)
            return minDistance

    # the minimum distance between point and one of the point in point1s
    def chipDistanceH(self, point, point1s, game_state):
      
        myqueue = PriorityQueue()
        startCost = 0
        parentPoint = (-1, -1)
        startNode = (parentPoint, point, startCost)
        # TODO: push the initial node into the queue, with F(n)
        myqueue.push(startNode, startCost)
        # create a visited-node set
        visited = set()
        # a dict to store the best g_cost
        best_g = {}
        totalCost = 0
        expandTime = 0

        while myqueue:
            node = myqueue.pop()
            expandTime += 1
            parent, pos, cost = node
            # take out the best cost for current pos
            best = best_g.setdefault(pos, cost)

            # check if the node has been visited, or need to reopen
            if pos not in visited or cost < best:
                best_g.update({pos: cost})
                if pos in point1s:
                    totalCost = cost
                    break

                # if it's the first time to explore, then 8 positions around it
                # should be expanded
                if expandTime == 1:
                    succNodes = self.expandH(parent, pos, game_state, True)
                else:
                    # if not, only need to expand in one direction
                    succNodes = self.expandH(parent, pos, game_state, False)

                if succNodes == []:
                    succNodes = self.expandA(pos)
                for succNode in succNodes:
                    succParent, succPos, succCost = succNode
                    newNode = (succParent, succPos, cost + succCost)
                    myqueue.push(newNode, cost + succCost)
        return totalCost

        # used to expand the node

    def expandH(self, parentPoint, point, game_state, isRoot):

        # chips is used to find it expand position contains opponent
        chips = game_state.board.chips
        opp_color = game_state.agents[self.id].opp_colour
        opp_seq_color = game_state.agents[self.id].opp_seq_colour

        # point's parent
        x0, y0 = parentPoint
        # current node which is going to be expanded
        x, y = point

        # the 4 hearts and four corner
        awesomeSet = set()
        awesomeSet.update(((4, 4), (4, 5), (5, 4), (5, 5), (0, 0), (0, 9), (9, 9), (9, 0)))

        # expand 8 directions
        if isRoot:
            smallSeqs = [[(-1, 0), (0, 0), (1, 0)],
                         [(0, -1), (0, 0), (0, 1)],
                         [(-1, -1), (0, 0), (1, 1)],
                         [(-1, 1), (0, 0), (1, -1)]]
            children = []
            for seq in smallSeqs:
                for i in range(len(seq)):
                    dx, dy = seq[i]
                    if x + dx >= 0 and x + dx < 10 and y + dy >= 0 and y + dy < 10:
                        if not (dx == dy and dx == 0):
                            # if opp is in the way, increase the cost
                            if chips[x + dx][y + dy] == opp_color:
                                children.append((point, (x + dx, y + dy), 1))
                            elif chips[x + dx][y + dy] == opp_seq_color:
                                children.append((point, (x + dx, y + dy), 1))
                            # if aswesome set we want get closer, so reduce the cost
                            elif chips[x + dx][y + dy] == EMPTY and (x + dx, y + dy) in awesomeSet:
                                children.append((point, (x + dx, y + dy), 1))
                            else:
                                children.append((point, (x + dx, y + dy), 1))
        else:
            # only expand one direction
            children = []
            dx = x - x0
            dy = y - y0
            if x + dx >= 0 and x + dx < 10 and y + dy >= 0 and y + dy < 10:
                # if opp is in the way, increase the cost
                if chips[x + dx][y + dy] == opp_color:
                    children.append((point, (x + dx, y + dy), 1))
                elif chips[x + dx][y + dy] == opp_seq_color:
                    children.append((point, (x + dx, y + dy), 1))
                # if aswesome set we want get closer, so reduce the cost
                elif chips[x + dx][y + dy] == EMPTY and (x + dx, y + dy) in awesomeSet:
                    children.append((point, (x + dx, y + dy), 1))
                else:
                    children.append((point, (x + dx, y + dy), 1))
        return children

    # expand the eight node around current position
    def expandA(self, point):
        x, y = point
        smallSeqs = [[(-1, 0), (0, 0), (1, 0)],
                     [(0, -1), (0, 0), (0, 1)],
                     [(-1, -1), (0, 0), (1, 1)],
                     [(-1, 1), (0, 0), (1, -1)]]
        children = []
        for seq in smallSeqs:
            for i in range(len(seq)):
                dx, dy = seq[i]
                if x + dx >= 0 and x + dx < 10 and y + dy >= 0 and y + dy < 10:
                    if not (dx == dy and dx == 0):
                        children.append((point, (x + dx, y + dy), 1))
        return children


####################################################################################################
class PriorityQueue:
    """
      Lowest cost priority queue data structure.
    """

    def __init__(self):
        self.heap = []
        self.count = 0

    def push(self, item, priority):
        entry = (priority, self.count, item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.heap)
        return item

    def isEmpty(self):
        return len(self.heap) == 0

    def update(self, item, priority):
        # If item already in priority queue with higher priority, update its priority and rebuild the heap.
        # If item already in priority queue with equal or lower priority, do nothing.
        # If item not in priority queue, do the same thing as self.push.
        for index, (p, c, i) in enumerate(self.heap):
            if i == item:
                if p <= priority:
                    break
                del self.heap[index]
                self.heap.append((priority, c, item))
                heapq.heapify(self.heap)
                break
        else:
            self.push(item, priority)


class AdvancedDict(dict):
    """
    A Dictionary with initial 0.0 if none key is got, and provide
    dot product of two dictionaries
    """

    def __getitem__(self, idx):
        self.setdefault(idx, 0.0)
        return dict.__getitem__(self, idx)

    def argMax(self):
        """
        Returns the key with the highest value.
        """
        if len(self.keys()) == 0: return None
        all = self.items()
        values = [x[1] for x in all]
        maxIndex = values.index(max(values))
        return all[maxIndex][0]

    def __mul__(self, y):
        sum = 0
        x = self
        if len(x) > len(y):
            x, y = y, x
        for key in x:
            if key not in y:
                continue
            sum += x[key] * y[key]
        return sum
