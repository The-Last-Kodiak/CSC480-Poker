#!/usr/bin/env python3
import random
import time
import math
import itertools
import argparse
from collections import Counter

# -----------------------------
# Card Representation & Parsing
# -----------------------------
RANK_SYMBOLS = '23456789TJQKA'
SUIT_SYMBOLS = 'cdhs'

def parse_card(s: str) -> int:
    r = RANK_SYMBOLS.index(s[0].upper())
    suit = SUIT_SYMBOLS.index(s[1].lower())
    return suit * 13 + r

def card_str(card: int) -> str:
    r = card % 13
    s = card // 13
    return RANK_SYMBOLS[r] + SUIT_SYMBOLS[s]

# -----------------------------
# Deck Management
# -----------------------------
class Deck:
    def __init__(self, exclude=None):
        self.cards = list(range(52))
        if exclude:
            for c in exclude:
                if c in self.cards:
                    self.cards.remove(c)
        random.shuffle(self.cards)

    def draw(self, n=1):
        if n <= 0:
            return []
        drawn = self.cards[-n:]
        self.cards[-n:] = []
        return drawn

# -----------------------------
# Hand Evaluation Helpers
# -----------------------------
def find_best_straight(ranks: list[int]) -> list[int] | None:
    uniq = sorted(set(ranks), reverse=True)
    if 12 in uniq:
        uniq.append(-1)
    for i in range(len(uniq)):
        run = [uniq[i]]
        for r in uniq[i+1:]:
            if r == run[-1] - 1:
                run.append(r)
            elif r < run[-1] - 1:
                break
        if len(run) >= 5:
            return run[:5]
    return None

# -----------------------------
# Full Hand Evaluator
# -----------------------------
def evaluate_hand(cards: list[int]) -> tuple:
    ranks = [c % 13 for c in cards]
    suits = [c // 13 for c in cards]
    counts = Counter(ranks)
    quads = [r for r,c in counts.items() if c == 4]
    trips = [r for r,c in counts.items() if c == 3]
    pairs = sorted([r for r,c in counts.items() if c == 2], reverse=True)
    for suit in range(4):
        suited = [r for r,s in zip(ranks, suits) if s == suit]
        if len(suited) >= 5:
            fb = sorted(suited, reverse=True)
            sf = find_best_straight(fb)
            if sf:
                return (8, *sf)
            return (5, *fb[:5])
    st = find_best_straight(ranks)
    if st:
        return (4, *st)
    if quads:
        q = quads[0]
        k = max(r for r in ranks if r != q)
        return (7, q, k)
    if trips and (len(trips) > 1 or pairs):
        t = trips[0]
        p = trips[1] if len(trips) > 1 else pairs[0]
        return (6, t, p)
    if trips:
        t = trips[0]
        ks = sorted((r for r in ranks if r != t), reverse=True)[:2]
        return (3, t, *ks)
    if len(pairs) >= 2:
        p1, p2 = pairs[:2]
        k = max(r for r in ranks if r not in (p1, p2))
        return (2, p1, p2, k)
    if pairs:
        p = pairs[0]
        ks = sorted((r for r in ranks if r != p), reverse=True)[:3]
        return (1, p, *ks)
    hc = sorted(ranks, reverse=True)[:5]
    return (0, *hc)

# -----------------------------
# MCTS Node & Search
# -----------------------------
class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children: list[MCTSNode] = []
        self.wins = 0
        self.visits = 0
        self.untried_actions: list[tuple[int,int]] = []
    def ucb1(self, total_visits, c=1.41) -> float:
        if self.visits == 0:
            return float('inf')
        return (self.wins/self.visits) + c*math.sqrt(math.log(total_visits)/self.visits)

class PokerMCTS:
    def __init__(self, hole_cards, community_cards, time_limit=10.0):
        self.hole = tuple(hole_cards)
        self.community = tuple(community_cards)
        self.time_limit = time_limit
        self.root = MCTSNode((self.hole, self.community, None))
        deck0 = Deck(exclude=list(self.hole)+list(self.community))
        self.root.untried_actions = list(itertools.combinations(deck0.cards.copy(),2))
    def run(self) -> float:
        start = time.time()
        while time.time() - start < self.time_limit:
            node = self._select(self.root)
            if node.untried_actions:
                node = self._expand(node)
            res = self._simulate(node)
            self._backprop(node,res)
        return self.root.wins/self.root.visits if self.root.visits else 0.0
    def _select(self,node):
        while not node.untried_actions and node.children:
            node = max(node.children,key=lambda n: n.ucb1(node.visits))
        return node
    def _expand(self,node):
        opp_hole = node.untried_actions.pop()
        child = MCTSNode((self.hole,self.community,opp_hole),parent=node)
        node.children.append(child)
        return child
    def _simulate(self,node):
        hole,community,opp_hole = node.state
        known = list(hole)+list(community)+list(opp_hole)
        deck = Deck(exclude=known)
        needed = 5-len(community)
        final_comm = list(community)+deck.draw(needed)
        ours = evaluate_hand(list(hole)+final_comm)
        opps = evaluate_hand(list(opp_hole)+final_comm)
        return 1 if ours>opps else 0
    def _backprop(self,node,res):
        while node:
            node.visits+=1
            node.wins+=res
            node=node.parent

# -----------------------------
# Decision Function
# -----------------------------
def decide_fold_or_stay(hole_cards, community_cards, time_limit=10.0) -> str:
    mcts = PokerMCTS(hole_cards,community_cards,time_limit)
    wp = mcts.run()
    return 'stay' if wp>=0.5 else 'fold'

# -----------------------------
# Command-line Driver (Testing)
# -----------------------------
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--hole',nargs=2,required=True)
    parser.add_argument('--community',nargs='*',default=[])
    parser.add_argument('--time',type=float,default=10.0)
    args=parser.parse_args()
    hole=[parse_card(c) for c in args.hole]
    community=[parse_card(c) for c in args.community]
    dec_pf=decide_fold_or_stay(hole,community,args.time)
    print('Pre-flop decision:',dec_pf)
    if dec_pf=='fold': exit(0)
    deck=Deck(exclude=hole+community)
    flop=deck.draw(3)
    print('Flop:',[card_str(c) for c in flop])
    dec_f=decide_fold_or_stay(hole,flop,args.time)
    print('Pre-turn decision:',dec_f)
    if dec_f=='fold': exit(0)
    turn=deck.draw(1)
    print('Turn:',card_str(turn[0]))
    community_t=flop+turn
    dec_t=decide_fold_or_stay(hole,community_t,args.time)
    print('Pre-river decision:',dec_t)
    if dec_t=='fold': exit(0)
    river=deck.draw(1)
    print('River:',card_str(river[0]))
    community_r=community_t+river
    mcts_wp=PokerMCTS(hole,community_r,args.time).run()
    print(f'MCTS win probability: {mcts_wp:.3f}')
