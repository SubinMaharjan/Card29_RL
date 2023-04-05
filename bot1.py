from utils import get_suit, get_suit_card
# from playCard29 import get_card, get_bid_limit_and_suit_card
from nn_game import get_card


# def get_bid(body):
#     """
#     Please note: this is bare implementation of the bid function.
#     Do make changes to this function to throw valid bid according to the context of the game.
#     """
    
#     ####################################
#     #     Input your code here.        #
#     ####################################
    
#     # MIN_BID = 16
#     # PASS_BID = 0
    
    

#     # # when you are the first player to bid, use the minimum bid
#     # if len(body["bidHistory"]) == 0:
#     #     return {"bid" : MIN_BID}

#     # return {"bid" : PASS_BID}

#     MIN_BID = 16
#     PASS_BID = 0
#     MY_BID = 0
#     MY_BID_LIMIT = 0

#     playerId = body["playerId"]
#     playerIds = body["playerIds"]
    

#     if len(body["bidHistory"]) == 0:
#         return {"bid" : MIN_BID}

#     ### Limit the bid based on the number of same suit card
#     ### IF needed later, need to change based on the value card
#     # max_suit_card, number_of_max_suit_card = get_max_suit_card(body["cards"])
#     # if number_of_max_suit_card < 2: MY_BID_LIMIT = 0
#     # elif number_of_max_suit_card < 3: MY_BID_LIMIT = 18
#     # elif number_of_max_suit_card < 4: MY_BID_LIMIT = 20
#     # else: MY_BID_LIMIT = 22

#     ### GET BID LIMIT VALUE

#     MY_BID_LIMIT = get_bid_limit(body["cards"])

#     if body["bidState"]["defenderId"] == playerId: ### For defender the bid can be same as the challenger
#         MY_BID = body['bidState']['challengerBid']
#         # MY_BID = body["bidHistory"][-1][1]
#     else: ### For challenger the bid must be greater than the defender
#         MY_BID = body["bidState"]["defenderBid"]+1
#             # MY_BID = body["bidHistory"][-1][1]+1 

#     # if body["bidState"]["defenderId"] == playerIds[(playerIds.index(playerId)+2)%4] and MY_BID_LIMIT < 19: MY_BID = PASS_BID   
#     if MY_BID < MIN_BID: MY_BID = PASS_BID

#     if MY_BID > MY_BID_LIMIT: MY_BID = PASS_BID ### if set bid is greater than bid_limit, pass 
#     # if MY_BID == PASS_BID: MY_BID = PASS_BID    ### else set the bid
#     # print("playerId: {}   Bid: {}".format(playerId, MY_BID_LIMIT))
#     # print("MY BID: ", MY_BID)
#     return {"bid" : MY_BID}
    






# def get_trump_suit( body):
#     """
#     Please note: this is bare implementation of the chooseTrump function.
#     Do make changes to this function to throw valid card according to the context of the game.
#     """
    
    
#     ####################################
#     #     Input your code here.        #
#     ####################################
    
#     # last_card = body['cards'][-1]
#     # last_card_suit = get_suit(last_card)
#     # return {"suit": last_card_suit}

#     # max_suit_card, number_of_max_suit_card = get_max_suit_card(body["cards"])
#     # print(game.trump)
#     trump_suit = get_trump_limit(body["cards"])
    
#     # trump_suit, _ = get_bid_limit_and_suit_card(body["cards"])
#     return {"suit": trump_suit}
    
    






def get_play_card(body, model_eval):
    """
    Please note: this is bare implemenation of the play function.
    It just returns the last card that we have.
    Do make changes to the function to throw valid card according to the context of the game.
    """
    
    ####################################
    #     Input your code here.        #
    ####################################
    
    own_cards = body["cards"]
    first_card = None if len(body["played"]) == 0 else body["played"][0]
    trump_suit = body["trumpSuit"]
    trump_revealed = body["trumpRevealed"]
    hand_history = body["handsHistory"]
    player_id = body["playerId"]
    player_number = body["playerIds"].index(player_id)

    played_cards = body["played"]

    # if we are the one to throw the first card in the hands
    if(not first_card):
        # return {"card": own_cards[-1]}
        return {"card": get_card(own_cards, trump_suit, hand_history, player_number,[], model_eval)}
        
    
    first_card_suit = get_suit(first_card)
    own_suit_cards = get_suit_card(own_cards, first_card_suit)
    
    # if we have the suit with respect to the first card, we throw it
    if len(own_suit_cards) > 0:
        # return {"card": own_suit_cards[-1]}
        return {"card" : get_card(own_cards, trump_suit, hand_history, player_number, played_cards, model_eval)}

    
    
    # if we don't have cards that follow the suit
    # @example
    # the first player played "7H" (7 of hearts)
    # 
    # we could either
    #
    # 1. throw any card
    # 2. reveal the trump 
    
    
    # trump suit is already revealed, and everyone knows the trump
    if (trump_suit and trump_revealed):
        was_trump_revealed_in_this_round = trump_revealed["hand"] == len(hand_history) + 1
        did_i_reveal_the_trump = trump_revealed["playerId"] == player_id
    
        # if I'm the one who revealed the trump in this round
        if was_trump_revealed_in_this_round and did_i_reveal_the_trump:
            trump_suit_cards = get_suit_card(own_cards, trump_suit)
            
            # player who revealed the trump should throw the trump suit card 
            if len(trump_suit_cards) > 0:
                # return {"card": trump_suit_cards[-1]}
                return {"card": get_card(own_cards, trump_suit, hand_history, player_number, played_cards, model_eval)}
            
        # if trump suit card is not available, throw any card
        # return {"card": own_cards[-1]}
        return {"card": get_card(own_cards, trump_suit, hand_history, player_number, played_cards, model_eval)}
        
    
    
    # trump is revealed only to me
    # this means we won the bidding phase, and set the trump
    if (trump_suit and not trump_revealed):
        trump_suit_cards = get_suit_card(own_cards, trump_suit)
        
        # after revealing the trump, we should throw the trump suit card if we have one
        if len(trump_suit_cards) > 0:
            return {
                "revealTrump": True,
                # "card": trump_suit_cards[-1]
                "card": get_card(own_cards, trump_suit, hand_history, player_number, played_cards, model_eval)
            }
        
        else:
            return {
                "revealTrump": True,
                # "card": own_cards[-1]
                "card": get_card(own_cards, trump_suit, hand_history, player_number, played_cards, model_eval)
            }
    
    # trump has not yet been revealed, let's reveal the trump
    return {"revealTrump" : True}

# from multiprocessing import Pool

# if __name__ == "__main__":
#     with Pool(5) as pool:
#         payload = {"playerId":"0DUt9Mq0","playerIds":["tgygXxAs","Bot 0","0DUt9Mq0","Bot 1"],"cards":["JS", "TS", "KH", "9S"],"timeRemaining":88.83185900000015,"bidHistory":[["Bot 0",16],["0DUt9Mq0",0],["Bot 1",0],["tgygXxAs",17],["Bot 0",17],["tgygXxAs",0]],"played":["JC"],"teams":[{"players":["tgygXxAs","0DUt9Mq0"],"bid":0,"won":10},{"players":["Bot 0","Bot 1"],"bid":17,"won":11}],"handsHistory":[["Bot 0",["JH","7H","QH","KH"],"Bot 0"],["Bot 0",["JD","7D","QD","8D"],"Bot 0"],["Bot 0",["8H","KD","8C","9H"],"tgygXxAs"],["tgygXxAs",["JS","8S","TS","KS"],"tgygXxAs"],["tgygXxAs",["7S","TD","9S","1S"],"0DUt9Mq0"],["0DUt9Mq0",["1D","TC","9D","TH"],"Bot 0"]],"trumpSuit":"H","trumpRevealed":{"hand":3,"playerId":"0DUt9Mq0"}}
#         print(game.get_bid_limit_and_suit_card( payload["cards"]))