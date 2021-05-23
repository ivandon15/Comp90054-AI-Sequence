from numpy import sign
from template import Agent
import heapq
from Sequence.sequence_model import *

"""
Authors: Group-28 Unimelb comp90054 2021s1
"""
EPSILON = 0.05
GAMMA = 0.9
ALPHA = 0.001

class myAgent(Agent):
    def __init__(self, _id):
        super().__init__(_id)

#         self.draft_weight = {'draft-take-two-eyed': 177.69943703971265, 'draft-take-one-eyed': 95.13716326170085, 'draft-seq-num': 43.63648248423927, 'draft-chip-num': 14.76295710890952}
#         self.remove_weight ={'remove-hearts': 1071.17265270355816, 'remove-seq-num': 48.64349546474051, 'remove-chip-num': -1.0161758953786348, 'remove-opp-chip-num': 1.3113270145486915}
#         self.play_weight = {'play-hearts': 115.220117390812621, 'play-seq-num': 94.47075623726205, 'play-chip-num': 28.565343263978622, 'play-opp-seq-num': 50.0, 'play-opp-chip-num': 10.592611505323564}
        self.draft_weight = AdvancedDict()
        self.remove_weight = AdvancedDict()
        self.play_weight = AdvancedDict()

    def SelectAction(self, actions, game_state):
        count = 1
        for line in open("QlearnWeight.txt", "r"):
            line = line.strip()
            if count == 1:
                self.draft_weight = eval(line)
                count += 1
            elif count == 2:
                self.remove_weight = eval(line
                count += 1
            else:
                self.play_weight = eval(line)
                
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
                     last_game_state.agents[self.id].score + \
                     game_state.agents[(self.id + 2) % 4].score - \
                     last_game_state.agents[(self.id + 2) % 4].score
            penalize = game_state.agents[(self.id + 1) % 4].score - \
                       last_game_state.agents[(self.id + 1) % 4].score+ \
                       game_state.agents[(self.id + 3) % 4].score - \
                       last_game_state.agents[(self.id + 3) % 4].score
            self.obeserveTransition(self.lastState, self.lastAction, whole_state, reward - penalize)
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
            print("updateQvalue-remove")
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
        f = open("QlearnWeight.txt", 'w')
        f.write(str(self.draft_weight))
        f.write("\n")
        f.write(str(self.remove_weight))
        f.write("\n")
        f.write(str(self.play_weight))
        f.write("\n")
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
                print("try to call checkSeq")
                seq_info = self.checkSeq(chips, plr_state, (x, y))
                print("calling checkSeq success and info is:",seq_info)
                chips[x][y] = EMPTY
                feature["draft-seq-num"],feature["draft-chip-num"] = seq_info
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
        opp_plr_state = game_state.agents[(self.id + 1) % 4]
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
        feature["remove-seq-num"],feature["remove-chip-num"] = seq_info
        chips[x][y] = opp_color
        seq_info = self.checkSeq(chips, opp_plr_state, (x, y))
        chips[x][y] = EMPTY
        temp, feature["remove-opp-chip-num"] = seq_info
        print("remove-opp-chip-num",feature["remove-opp-chip-num"])
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
        print("opp_plr:",opp_plr_state)
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
        feature["play-seq-num"],feature["play-chip-num"] = seq_info
        chips[x][y] = opp_color
        seq_info = self.checkSeq(chips, opp_plr_state, (x, y))
        chips[x][y] = EMPTY
        feature["play-opp-seq-num"],feature["play-opp-chip-num"] = seq_info
        print("play feature",feature)
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

    ## for sequence

    def register(self, whole_state):
        self.startEpoch()

    def final(self, whole_state):
        game_state = whole_state[0]
        last_game_state = self.lastState[0]
        deltaReward = game_state.agents[self.id].score - \
                      last_game_state.agents[self.id].score
        self.obeserveTransition(self.lastState, self.lastAction, whole_state, deltaReward)
        # self.endEpoch()

    ######################################################33
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

        chip_count = 0

        for seq, seq_name in [(vr, 'vr'), (hz, 'hz'), (d1, 'd1'), (d2, 'd2')]:
            coord_list = [(r + lr, c + lc) for r, c in seq]
            coord_list = [i for i in coord_list if 0 <= min(i) and 9 >= max(i)]  # Sequences must stay on the board.
            chip_str = ''.join([chips[r][c] for r, c in coord_list])

            for chip_chr in chip_str:
                if chip_chr == clr or chip_chr == sclr:
                    chip_count += 1

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
        print("checkseq-num",chip_count)
        return (num_seq, chip_count)

    def oppAlmostSeq(self, position, chips, opp_color):
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
