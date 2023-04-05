# from bot1 import get_play_card
from nn_game import format_card, format_trump
from copy import deepcopy
import random
import json
from reward_utils import *
import numpy as np
import os

def get_deck():
    cardFaces = 'J91TKQ87'
    suits = 'HDCS'
    deck = [cardface + suit for cardface in cardFaces for suit in suits]
    return deck

class Player:
    def __init__(self, privateCards, trump=False):
        # Trump is either false or any suit from trump suits
        self.privateCards = privateCards
        self.trump_suit = trump
    
    def set_trump(self, trump_suit):
        self.trump_suit = trump_suit

    def remove_card(self, card):
        self.privateCards.remove(card)
    
def add_public_card(publicCards, cards):
    publicCards.extend(cards)
    return publicCards

def initialize_player(cards):
    players = [None]*4
    for i, card in enumerate(cards):
        #Initialize all players
        players[i] = Player(card)
    return players

def get_next_player(playerToMove):
    return ((playerToMove+1)%4)

def calculate_rewards(winner, played_cards, points_to_dict):
    rank, points = '78QKT19J', '00001123'
    rewards = [0]*4
    # suitedPlays = [card for card in played_cards if card[1] == played_cards[0][1]]
    # trumpPlays = [card for card in played_cards if card[1] == trump_suit]
    # sortedPlays = sorted(suitedPlays, key=lambda card: rank.index(card[0])) + sorted(trumpPlays, key=lambda card: rank.index(card[0]))
    # winner = (first_player+played_cards.index(sortedPlays[-1])) % 4
    points = sum([int(points[rank.index(card[0])]) for card in played_cards])
    rewards[winner], rewards[(winner+2)%4] = points, points * 0.90 # Teammate gets 90% of the total point
    #Add reward to points dictionary
    points_to_dict[winner] += rewards[winner]
    points_to_dict[winner] += rewards[(winner+2)%4]

    # print("Winner: {}, Rewards: {}".format(winner, rewards))
    # print("Reward according to first player: [{} {} {} {}]".format(*[rewards[(first_player+i)%4] for i in range(4)]))
    return points_to_dict


def GenerateData(playerIds, cards, handsHistory, trump, trumpRevealRound, trumpRevealPlayerId):
    players = initialize_player(cards)
    players[trumpRevealPlayerId].set_trump(trump)
    states, trumpInfo, publicCards,playStyle,rewards = [], [], [], [] ,[]
    for handHistory in handsHistory:
        playedCards = []
        points_dict = {0: 0, 1: 0, 2: 0, 3: 0}
        firstPlayer = playerIds.index(handHistory[0])
        playStyle.append([(firstPlayer+i)%4 for i in range(4)])
        winner = playerIds.index(handHistory[2])
        CardsInTrick = handHistory[1]
        playerToMove = firstPlayer
        for round, card in enumerate(CardsInTrick):
            if trumpRevealRound == round+1 and playerToMove == trumpRevealPlayerId:
                for i in range(4):
                    players[i].set_trump(trump)
            players[playerToMove].remove_card(card)
            states.append(format_card(deepcopy(players[playerToMove].privateCards),deepcopy(publicCards), deepcopy(playedCards)))
            trumpInfo.append(format_trump(deepcopy(players[playerToMove].trump_suit)))
            playedCards.append(card)
            playerToMove = get_next_player(playerToMove)
        publicCards = add_public_card(publicCards, playedCards)
        points_dict = calculate_rewards(winner, playedCards, points_dict)
        rewards = points_dict_to_reward_list(points_dict, rewards)
    G = calculate_G(rewards)
    playStyle = np.asarray(playStyle)
    G = np.array(list(map(lambda x, y: y[x], playStyle, G)))
    final_G = np.reshape(G, (32,1)) 
    return states, trumpInfo, final_G


def read_data(folderpath):
    final_states, final_trumpInfo, final_rewards = [], [], []
    # Opening JSON file
    for i, file in enumerate(os.listdir(folderpath)):
        print("FileName: ", file)
        f = open(folderpath + file)
        # returns JSON object as 
        # a dictionary
        data = json.load(f)
        
        for idx, play in enumerate(data["games"]):
            if (idx+1)% 100 == 0: print("Number of plays: ", idx+1)
            playerIds = play["playerIds"]
            cards = play["cards"]
            handsHistory = play["handsHistory"]["handsHistory"]
            trump = play['trump']
            if play["trumpReveal"] != {}:
                trumpRevealRound = play["trumpReveal"]["round"]
                trumpRevealPlayerId = play["trumpReveal"]["playerIdx"]
            else:
                trumpRevealRound = -1
                trumpRevealPlayerId = -1
            states, trumpInfo, rewards = GenerateData(playerIds, cards, handsHistory, trump, trumpRevealRound, trumpRevealPlayerId)
            final_states.extend(states)
            final_trumpInfo.extend(trumpInfo)        
            final_rewards.extend(rewards)
    
    return np.array(final_states), np.array(final_trumpInfo), np.array(final_rewards)


if __name__ == "__main__":
    states, trumpInfo, rewards = read_data("./data/")
    print(states.shape)
    np.savez("datasets/data0.npz", states = states, trumpInfo = trumpInfo, rewards = rewards)