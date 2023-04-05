import numpy as np
# from train import get_model
# from tensorflow.keras.models import load_model
import pickle
import joblib

# model = get_model()
# model.load_weights('compressed_model_normalized.h5')
# model = load_model('rl/compressed_model_normalized.h5')
# bid_model = load_model('bid_model_normalized.h5')
# trump_model = load_model('trump_model_normalized.h5')

model = joblib.load('model/model0.pkl')
# model = pickle.load(open('model/model0.pkl', 'rb'))
print(model)
print("ok")
faceIndex = 'J91TKQ87'
suitIndex = 'HDSC'
epsilon = 0.2



def get_bid_limit_and_suit_card(cards):
    """
        This function takes the initial 4 cards and separates max suit card,
        count number of max suit card
        number of Jack (high card) and 
        set the max limit of the bid
    """
    count_J = 0
    suits = {
        "H" : 0,
        "C" : 0,
        "S" : 0,
        "D" : 0
    }
    for card in cards:
        if card[0] == 'J': 
            count_J += 1
        suits[card[1]] += 1 

    
    count = 0
    trump_suit = None
    for suit in suits:
        if suits[suit] > count:
            trump_suit = suit
            count = suits[suit]
        
    bid_limit = 16    
    if count_J:
        bid_limit += count_J
    if count > 2:
        bid_limit += count-1
    if bid_limit == 16: bid_limit = 0
    # print("Bid_limit: ", bid_limit)
    return trump_suit, bid_limit


def store_all_played_cards(hand_history):
    all_played_cards = set()                    
    if hand_history == []: 
        return all_played_cards
    # if hand_history[-1][1][0][0] in all_played_cards[card[1]]: return
    all_played_cards = {card for idx in range(len(hand_history)) for card in hand_history[idx]}
    # for idx in range(len(hand_history)):
    #     for card in hand_history[idx][1]:
    #         all_played_cards.add(card)
    return all_played_cards

# def format_card(cards):
#     cardInarray = np.zeros((8,4))
#     for card in cards:
#         cardInarray[faceIndex.index(card[0]), suitIndex.index(card[1])] = 1
#     return cardInarray

def format_card(privateCards, publicCards, playedCards):
    cardInarray = np.zeros((8,4))
    # value_card = {'privateCards': 3, 'publicCards':1, 'playedCards':2}
    for card in privateCards:
        cardInarray[faceIndex.index(card[0]), suitIndex.index(card[1])] = 3
    for card in publicCards:
        cardInarray[faceIndex.index(card[0]), suitIndex.index(card[1])] = 1
    for card in playedCards:
        cardInarray[faceIndex.index(card[0]), suitIndex.index(card[1])] = 2
    return cardInarray

def format_trump(trumpSuit):
    suitInarray = np.zeros((5,1))
    if trumpSuit: 
        suitInarray[suitIndex.index(trumpSuit)] = 1
    else: suitInarray[4] = 1
    return suitInarray

def get_legal_cards(privateCards, played_cards, trumpSuit):
    rankIndex = 'J91TKQ87'
    if played_cards == []:
        # May lead a trick with any card
        return privateCards
    else:
        leadCard = played_cards[0]
        # Must follow suit if it is possible to do so
        cardsInSuit = [card for card in privateCards if card[1] == leadCard[1]]
        if cardsInSuit != []:
            return cardsInSuit
        elif trumpSuit != False:
            cardInTrumSuit = [card for card in privateCards if card[1] == trumpSuit]
            if cardInTrumSuit != []:
                if len(cardInTrumSuit) > 1:
                    played_trump_card = [card for card in played_cards if card[1] == trumpSuit]
                    usable_trump_card = [card for card in cardInTrumSuit for played_trump in played_trump_card if rankIndex.index(card[0]) > rankIndex.index(played_trump[0])]
                    if usable_trump_card != []: #and played_cards[-1][1] == trumpSuit:
                        return usable_trump_card
                    # elif played_trump_card == []: return cardInTrumSuit
                # else: return privateCards
                return cardInTrumSuit
        else:
            # Can't follow suit, so can play any card
            return privateCards
    return privateCards

def playGame(privateCards, publicCards, trumpSuit, played_cards, model_eval = None):
    legal_moves = get_legal_cards(privateCards, played_cards, trumpSuit)
    if len(legal_moves) == 1: return legal_moves[-1]
    # publicCards = format_card(publicCards).reshape(1,8,4)
    # played_cards = format_card(played_cards).reshape(1,8,4)
    trumpSuit = format_trump(trumpSuit).reshape(1,5,1)

    Q_state = {}
    for card in legal_moves:
        temp_privateCards = list(set(privateCards) - set(card))
        meta_card = format_card(temp_privateCards,publicCards,played_cards).reshape(1,8,4)
        if model_eval != None: 
            Q_state[card] = model_eval.predict([meta_card, trumpSuit], verbose = 0)
        else: Q_state[card] = model.predict([meta_card, trumpSuit], verbose = 0)
        best_action = max(Q_state.items(), key=lambda x: x[1])[0]
    if np.random.rand() > epsilon: #Exploitation    
        action = best_action
    else:
        action = np.random.choice(legal_moves) # Exploration
    return action
    

def get_card(own_cards, trump_suit, hand_history, player_number, played_cards=[], model_eval = None):
    privateCards = own_cards
    playerToMove = player_number
    publicCards = list(store_all_played_cards(hand_history))
    card = playGame(privateCards, publicCards, trump_suit, played_cards, model_eval)
    return card

def get_bid_limit(cards):
    cardInarray = np.zeros((8,4))
    for card in cards:
        cardInarray[faceIndex.index(card[0]), suitIndex.index(card[1])] = 1
    bid_limit = bid_model.predict(cardInarray.reshape(1,8,4), verbose = 0)
    return bid_limit

def get_trump_limit(cards):
    cardInarray = np.zeros((8,4))
    for card in cards:
        cardInarray[faceIndex.index(card[0]), suitIndex.index(card[1])] = 1
    trump = trump_model.predict(cardInarray.reshape(1,8,4), verbose = 0)
    idx = np.argmax(trump)
    if idx < len(suitIndex): return suitIndex[idx]
    idx = np.random.randint(0,3)
    return suitIndex[idx]
    