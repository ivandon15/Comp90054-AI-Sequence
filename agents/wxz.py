from template import Agent
from Sequence.sequence_model import BOARD, COORDS
from Sequence.sequence_utils import JOKER
import random

# CONSTANTS ----------------------------------------------------------------------------------------------------------#
HEURISTIC = [[100,3,4,4,4,4,4,4,3,100],
             [3,4,4,4,4,4,4,4,4,3],
             [4,4,4,4,4,4,4,4,4,4],
             [4,4,4,4,4,4,4,4,4,4],
             [4,4,4,4,0,0,4,4,4,4],
             [4,4,4,4,0,0,4,4,4,4],
             [4,4,4,4,4,4,4,4,4,4],
             [4,4,4,4,4,4,4,4,4,4],
             [3,4,4,4,4,4,4,4,4,3],
             [100,3,4,4,4,4,4,4,3,100]]


class myAgent(Agent):
    def __init__(self, _id):
        super().__init__(_id)

    # This method is used to calculate the heuristic value for each position on
    # the board
    def GetChipsHeuristic(self, colour, seq_us_colour, seq_opp_colour, chips):
        temp = [(i, j) for i in range(10) for j in range(10)]
        for (i,j) in temp:
            # ignore the position that already been occupied by us and opp_seq
            # or the JOKER
            if HEURISTIC[i][j] < 0.0001 or HEURISTIC[i][j] == 100 or chips[i][j] == \
                    colour or chips[i][j] == seq_us_colour or chips[i][j] == \
                    seq_opp_colour:
                temp.remove((i,j))

        # initial the heuristic
        chipsHeuristic = HEURISTIC
        for pair in temp:
            i,j = pair
            hori = 0
            vert = 0
            diag = 0
            cdia = 0
            for di in range(1,5):
                if i-di >= 0:
                    if chips[i - di][j] == JOKER or chips[i - di][j] == colour:
                        hori = hori + 1
                else:
                    break
            for di in range(1,5):
                if i+di <= 9:
                    if chips[i + di][j] == JOKER or chips[i + di][j] == colour:
                        hori = hori + 1
                else:
                    break
            for di in range(1,5):
                if j-di >= 0:
                    if chips[i][j - di] == JOKER or chips[i][j - di] == colour:
                        vert = vert + 1
                else:
                    break
            for di in range(1,5):
                if j+di < 10:
                    if chips[i][j + di] == JOKER or chips[i][j + di] == colour:
                        vert = vert + 1
                else:
                    break
            for di in range(1,5):
                if i-di >= 0 and j-di >= 0:
                    if chips[i - di][j - di] == JOKER or chips[i - di][j - di] == colour:
                        diag = diag + 1
                else:
                    break
            for di in range(1,5):
                if i+di <= 9 and j+di <= 9:
                    if chips[i + di][j + di] == JOKER or chips[i + di][j + di] == colour:
                        diag = diag + 1
                else:
                    break
            for di in range(1,5):
                if i+di <= 9 and j-di >= 0:
                    if chips[i + di][j - di] == JOKER or chips[i + di][j - di] == colour:
                        cdia = cdia + 1
                else:
                    break
            for di in range(1,5):
                if i-di >= 0 and j+di <= 9:
                    if chips[i - di][j + di] == JOKER or chips[i - di][j + di] == colour:
                       cdia = cdia + 1
                else:
                   break
            max_seq = max(hori,vert,diag,cdia)
            hvalue = max(4-max_seq, 0)
            chipsHeuristic[i][j] = hvalue

        return chipsHeuristic

    def GetCardHeuristic(self, chipsHeuristic):
        cards = [(r + s) for r in ['2', '3', '4', '5', '6', '7', '8', '9', 't', 'q', 'k', 'a'] for s in
                 ['d', 'c', 'h', 's']]
        cardsHeuristic = {i: 10000 for i in cards}
        cardsHeuristic["jd"] = 0
        cardsHeuristic["jc"] = 0
        cardsHeuristic["jh"] = 0
        cardsHeuristic["js"] = 0

        for card in cards:
            positions = COORDS[card]
            sumValue = 0
            for position in positions:
                x,y = position
                sumValue += chipsHeuristic[x][y]
            cardsHeuristic[card] = sumValue / len(positions)

        return cardsHeuristic

    def SelectAction(self, actions, game_state):
        # 首先，先计算棋盘的heuristic,和每一张牌的heuristic
        chipsH = self.GetChipsHeuristic(game_state.agents[self.id].colour, game_state.agents[self.id].seq_colour,
                                        game_state.agents[self.id].opp_seq_colour, game_state.board.chips)
        cardsH = self.GetCardHeuristic(chipsH)
        print(cardsH)

        # 以上，都是正确的
        # 如果必须换牌，就换成heuristic最低的牌
        if game_state.agents[self.id].trade:
            actionsList = []
            for action in actions:
                actionsList.append((cardsH[action["draft_card"]], action))
            actionsList.sort(key=lambda k:k[0])
            return actionsList[0][1]
        # 如果要出牌，就选择最小的heuristic地方操作，然后抽一张heuristic最小的牌
        else:
            actionsList = []
            for action in actions:
                x, y = action["coords"]
                actionsList.append((cardsH[action["draft_card"]] + chipsH[x][y], action))
            actionsList.sort(key=lambda k:k[0])
            return actionsList[0][1]