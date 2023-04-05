from bot1 import get_play_card
from nn_game import format_card, format_trump
from copy import deepcopy
import random
import json
from reward_utils import *
import numpy as np
from tensorflow.keras.models import load_model

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

def initialize_player(deck):
    number_of_players = 4
    players = [None]*4
    for i in range(number_of_players):
        #Initialize all players
        players[i] = Player(deck[:8])
        deck = deck[8:]
    return players

def get_next_player(playerToMove):
    return ((playerToMove+1)%4)

def calculate_rewards(first_player,played_cards, trump_suit, points_to_dict):
    rank, points = '78QKT19J', '00001123'
    rewards = [0]*4
    suitedPlays = [card for card in played_cards if card[1] == played_cards[0][1]]
    trumpPlays = [card for card in played_cards if card[1] == trump_suit]
    sortedPlays = sorted(suitedPlays, key=lambda card: rank.index(card[0])) + sorted(trumpPlays, key=lambda card: rank.index(card[0]))
    winner = (first_player+played_cards.index(sortedPlays[-1])) % 4
    points = sum([int(points[rank.index(card[0])]) for card in played_cards])
    rewards[winner], rewards[(winner+2)%4] = points, points * 0.90 # Teammate gets 90% of the total point
    #Add reward to points dictionary
    points_to_dict[winner] += rewards[winner]
    points_to_dict[winner] += rewards[(winner+2)%4]

    # print("Winner: {}, Rewards: {}".format(winner, rewards))
    # print("Reward according to first player: [{} {} {} {}]".format(*[rewards[(first_player+i)%4] for i in range(4)]))
    return winner, points_to_dict

def play_one_game(players, playerToMove, trump_suit, model_eval = None):
    no_of_tricks = 8 # One game has 8 rounds
    playerIds = [i for i in range(4)]
    trump_revealed = False
    trump_revealed_in_this_round = False
    hand_history = []
    publicCard = []
    # states = {'privateCards': [], 'publicCards': [], 'playedCards': [], 'trumpSuit': []}
    states, trumpInfo = [], []
    points_dict = {0: 0, 1: 0, 2: 0, 3: 0}
    rewards = []
    body = {'playerIds': playerIds, 'trumpRevealed': trump_revealed, 'handsHistory': hand_history,
     'played' : [], 'playerId': playerToMove, 'cards': players[playerToMove].privateCards, 'trumpSuit': players[playerToMove].trump_suit}
    play_style = []
    for i in range(no_of_tricks):
        
        played_cards = []
        first_player = playerToMove
        play_style.append([(first_player+i)%4 for i in range(4)])
        for j in range(4): # In each trick, 4 players throw the card
            # print("Private Card of Player {}: {}".format(playerToMove, players[playerToMove].privateCards))
            # print("Public cards: ", publicCard)
            # print("Played cards: ", played_cards)

            body['cards'] = players[playerToMove].privateCards
            body['trumpRevealed'] = trump_revealed
            body['handsHistory'] = hand_history
            body['played'] = played_cards
            body['playerId'] = playerToMove
            body['trumpSuit'] = players[playerToMove].trump_suit

            data = get_play_card(body, model_eval)
            card = data['card'] if 'card' in data else None
            if card == None and data['revealTrump'] == True:
                trump_revealed = {'hand': i+1, 'playerId': playerToMove}
                # print(trump_revealed)
                for i in range(4):
                    players[i].set_trump(trump_suit)
                body['trumpSuit'] = players[playerToMove].trump_suit
                data = get_play_card(body, model_eval)
                card = data['card']
            # Remove card from the players Hand
            players[playerToMove].remove_card(card)
            # Add card to public Card
            
            # Add to state
            # states['privateCards'].append(deepcopy(players[playerToMove].privateCards))
            # states['publicCards'].append(deepcopy(publicCard))
            # states['playedCards'].append(deepcopy(played_cards))
            # states['trumpSuit'].append(deepcopy(players[playerToMove].trump_suit))
            states.append(format_card(deepcopy(players[playerToMove].privateCards),deepcopy(publicCard), deepcopy(played_cards)))
            trumpInfo.append(format_trump(deepcopy(players[playerToMove].trump_suit)))
            played_cards.append(card)
            
            playerToMove = get_next_player(playerToMove)
        publicCard = add_public_card(publicCard, played_cards)
        winner, points_dict = calculate_rewards(first_player,played_cards, trump_suit, points_dict)
        rewards = points_dict_to_reward_list(points_dict, rewards)
        playerToMove = winner
        # print(reward)
        hand_history.append(played_cards)
    G = calculate_G(rewards, gamma=0.9)
    play_style = np.asarray(play_style)
    G = np.array(list(map(lambda x, y: y[x], play_style, G)))
    
    final_G = np.reshape(G, (32, 1))

    return states,trumpInfo, final_G


def play_one_game_eval(players, playerToMove, trump_suit, model_eval = None):
    no_of_tricks = 8 # One game has 8 rounds
    playerIds = [i for i in range(4)]
    trump_revealed = False
    trump_revealed_in_this_round = False
    hand_history = []
    publicCard = []
    # states = {'privateCards': [], 'publicCards': [], 'playedCards': [], 'trumpSuit': []}
    states, trumpInfo = [], []
    points_dict = {0: 0, 1: 0, 2: 0, 3: 0}
    rewards = []
    body = {'playerIds': playerIds, 'trumpRevealed': trump_revealed, 'handsHistory': hand_history,
     'played' : [], 'playerId': playerToMove, 'cards': players[playerToMove].privateCards, 'trumpSuit': players[playerToMove].trump_suit}
    play_style = []
    for i in range(no_of_tricks):
        
        played_cards = []
        first_player = playerToMove
        play_style.append([(first_player+i)%4 for i in range(4)])
        for j in range(4): # In each trick, 4 players throw the card
            # print("Private Card of Player {}: {}".format(playerToMove, players[playerToMove].privateCards))
            # print("Public cards: ", publicCard)
            # print("Played cards: ", played_cards)

            body['cards'] = players[playerToMove].privateCards
            body['trumpRevealed'] = trump_revealed
            body['handsHistory'] = hand_history
            body['played'] = played_cards
            body['playerId'] = playerToMove
            body['trumpSuit'] = players[playerToMove].trump_suit

            model_eval = model_eval if playerToMove % 2 == 0 else None

            data = get_play_card(body, model_eval)
            card = data['card'] if 'card' in data else None
            if card == None and data['revealTrump'] == True:
                trump_revealed = {'hand': i+1, 'playerId': playerToMove}
                # print(trump_revealed)
                for i in range(4):
                    players[i].set_trump(trump_suit)
                body['trumpSuit'] = players[playerToMove].trump_suit
                data = get_play_card(body, model_eval)
                card = data['card']
            # Remove card from the players Hand
            players[playerToMove].remove_card(card)
            states.append(format_card(deepcopy(players[playerToMove].privateCards),deepcopy(publicCard), deepcopy(played_cards)))
            trumpInfo.append(format_trump(deepcopy(players[playerToMove].trump_suit)))
            played_cards.append(card)
            
            playerToMove = get_next_player(playerToMove)
        publicCard = add_public_card(publicCard, played_cards)
        winner, points_dict = calculate_rewards(first_player,played_cards, trump_suit, points_dict)
        playerToMove = winner
        # print(reward)
        hand_history.append(played_cards)
    return winner, points_dict

def evaluate_model(model_eval = None, number_of_simulation = 1):
    # model_eval = load_model('compressed_model_normalized.h5')
    deck = get_deck()
    # Randomize deck
    random.shuffle(deck)
    ### A is model_eval B is model
    team_wins = {'A': 0, 'B': 0}

    for i in range(number_of_simulation):
        players = initialize_player(deck)
        playerToMove = random.randint(0,3) # Randomly select any players to be the first player
        playerToMove_suit = [card[1] for card in players[playerToMove].privateCards]
        trump_suit = random.choices(['H','D', 'C', 'S'], weights = [playerToMove_suit.count(suit) for suit in ['H','D', 'C', 'S']])[0]
        # print(trump_suit)
        players[playerToMove].set_trump(trump_suit)
        winner, points_dict = play_one_game_eval(players, playerToMove, trump_suit, model_eval)
        print(points_dict)
        player = max(points_dict, key = points_dict.get)
        if player % 2 ==0: team_wins['A'] += 1
        else: team_wins['B'] += 1
    
    return team_wins

        


def run_bid_simulation(number_of_simulation = 10):
    states = {'cards': [], 'bid': [], 'trump': []}
    deck = get_deck()
    for i in range(number_of_simulation):
        if (i+1) % 10 == 0: print("Number of game played: ", i+1)
        # Randomize deck
        random.shuffle(deck)
        game = bid_and_trump()
        playerToMove = random.randint(0,3) # Randomly select any players to be the first player
        cards = deck[:4]
        trump,bid = game.get_bid_limit_and_suit_card(cards, playerToMove)
        states['cards'].append(cards)
        states['bid'].append(bid)
        states['trump'].append(trump)
    return states


def run(number_of_simulation = 10):
    # final_states = {'privateCards': [], 'publicCards': [], 'playedCards': [], 'trumpSuit': []}
    final_states, final_trumpInfo = [], []
    final_rewards = []
    #Initially get a deck of card
    deck = get_deck()
    # Randomize deck
    random.shuffle(deck)
    prev= None
    for i in range(number_of_simulation):
        if (i+1) % 10 == 0: print("Number of game played: ", i+1)
        players = initialize_player(deck)
        playerToMove = random.randint(0,3) # Randomly select any players to be the first player
        playerToMove_suit = [card[1] for card in players[playerToMove].privateCards]
        trump_suit = random.choices(['H','D', 'C', 'S'], weights = [playerToMove_suit.count(suit) for suit in ['H','D', 'C', 'S']])[0]
        # print(trump_suit)
        players[playerToMove].set_trump(trump_suit)

        states, trumpInfo, rewards = play_one_game(players, playerToMove, trump_suit)
        # final_states['privateCards'].append(states['privateCards'])
        # final_states['publicCards'].append(states['publicCards'])
        # final_states['playedCards'].append(states['playedCards'])
        # final_states['trumpSuit'].append(states['trumpSuit'])
        final_states.extend(states)
        final_trumpInfo.extend(trumpInfo)        
        final_rewards.extend(rewards)
    return np.array(final_states), np.array(final_trumpInfo), np.array(final_rewards)

def train_model(model, states, trumpInfo, rewards):
    rewards =rewards/160 # normalized
    model.fit(
        x=[states, trumpInfo],
        y=rewards,
        batch_size=320,
        epochs=500,
        verbose=2,
        shuffle=True,
    )
    return model

def save_model(model):
    model.save('rl/compressed_model_normalized.h5')
    
        

if __name__ == "__main__":
    # deck = run()
    for i in range(5):
        states, trumpInfo, rewards = run(30)
        model = load_model('rl/compressed_model_normalized.h5')
        model_eval = train_model(model, states, trumpInfo,rewards)
        team_wins = evaluate_model(model_eval, number_of_simulation = 20)
        print(team_wins)
        if team_wins['A'] > team_wins['B']: 
            save_model(model_eval)
            print("model saved...")



    # states['rewards'] = rewards.to_list()
    # with open("data0.json", "w") as outfile:
    #     json.dump(states, outfile, indent=4)
    # np.save('reward_data.npy', rewards)
    
    ########################### For Bid and Trump Selection ##############################
    # states = run_bid_simulation(1000)
    # with open("bid_and_trump.json", "w") as outfile:
    #     json.dump(states, outfile, indent=4)
