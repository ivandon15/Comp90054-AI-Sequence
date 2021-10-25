from template import Agent
from Sequence.sequence_model import BOARD, COORDS
import random


class myAgent(Agent):
    def __init__(self,_id):
        super().__init__(_id)
    def findPos(self,card, actions):
        ans = set()
        for action in actions:
            if action['play_card'] == card:
                ans.add(action['coords'])
        return list(ans)
    
    def calValue(self,pos, colour, seq_colour, chips):
        seqs = [[(-4,0),(-3,0),(-2,0),(-1,0),(0,0),(1,0),(2,0),(3,0),(4,0)],
                 [(0,-4),(0,-3),(0,-2),(0,-1),(0,0),(0,1),(0,2),(0,3),(0,4)],
                 [(-4,-4),(-3,-3),(-2,-2),(-1,-1),(0,0),(1,1),(2,2),(3,3),(4,4)],
                 [(-4,4),(-3,3),(-2,2),(-1,1),(0,0),(1,-1),(2,-2),(3,-3),(4,-4)]]
        x, y = pos
        maxV = 0
        for seq in seqs:
            value = 0
            index = 0
            for i in range(len(seq)):
                dx,dy = seq[i]
                if x+dx>=0 and x+dx<10 and y+dy>=0 and y+dy<10:
#                    print(BOARD[x+dx][y+dy])
                    index += 1
                    if index > 5:
                        index = 5
                        ddx,ddy = seq[i-5]
                        if chips[x+ddx][y+ddy] == '#' or chips[x+ddx][y+ddy] == colour or chips[x+ddx][y+ddy] == seq_colour:
                            value -= 1
                    if chips[x+dx][y+dy] == '#' or chips[x+dx][y+dy] == colour or chips[x+dx][y+dy] == seq_colour:
                        value += 1
                    maxV = max(maxV,value)
        print(pos,maxV)
        return maxV
    def SelectAction(self,actions,game_state):
        action = actions[0]
#        print(game_state.agents[self.id].hand)
        if game_state.agents[self.id].trade:
            return random.choice(actions)
        else:
            maxV = 0
            bestCard = action['play_card']
            bestPos = action['coords']
            for card in game_state.agents[self.id].hand:
#                poss = COORDS[card]
#                p1, p2 = poss
#                x1, y1 = p1
#                print(card, poss)
#                print(game_state.board.chips[x1][y1])
                positions = self.findPos(card, actions)
                print(positions)
                for pos in positions:
#                    print(pos,game_state.agents[self.id].colour,game_state.agents[self.id].seq_colour,game_state.board.chips)
                    value = self.calValue(pos,game_state.agents[self.id].colour,game_state.agents[self.id].seq_colour,game_state.board.chips)
                    if value > maxV:
                        bestPos = pos
                        maxV = value
                        bestCard = card
            print(bestPos,bestCard,maxV)
            bestActions = []
            for action in actions:
                if action['play_card'] == bestCard and action['coords'] == bestPos:
                    bestActions.append(action)
            return random.choice(bestActions)

    