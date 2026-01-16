import random

def create_deck():
  suits = ["♥", "♦", "♣", "♠"]
  ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]

  deck = []

  for suit in suits:
    for rank in ranks:
      deck.append((suit, rank))
  random.shuffle(deck)
  
  return deck

def draw_card(deck, num_cards):
  hand = []
  for card in range(num_cards):
    if deck:
      hand.append(deck.pop(-1))
    else:
      break
  return hand, deck

deck = create_deck()

def show_card(card):
  space = " "
  if len(card[1]) == 2:
    space = ""
  print (f"""
+-------+a
|{card[1]}     {space}|
|       |
|   {card[0]}   |
|       |
|{space}     {card[1]}|
+-------+
""")
  print(f"There are {len(deck)} cards left in the deck")


while len(deck) > 0:
  num_cards = int(input("How many cards would you like to draw?"))
  hand, deck = draw_card(deck, num_cards)
  for card in hand:
    show_card(card)

print("We are out of card")
