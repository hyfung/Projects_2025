# Poker

## Probability Analysis

### Combination

- 2 Hand cards
- 5 River cards

Each player has 7 cards, the combination is given by $ C^{52}\_{7} = 133784560 $

| Number of River Cards | Combination              |
| --------------------- | ------------------------ |
| 3                     | $ C_3^{52} = 22100 $     |
| 4                     | $ C_4^{52} = 270725$     |
| 5                     | $ C_5^{52} = 2598960 $   |
| 6                     | $ C_6^{52} = 20358520 $  |
| 7                     | $ C_7^{52} = 133784560 $ |

### Value Precedence

- 5-card
  - Straight Flush
  - Four-of-a-kind
  - Full House
  - Flush
  - Straight
- 3-card
  - Three-of-a-kind
- 2-card
  - Two pairs
  - Pair

### Probability Calculation

Since the state in the system is finite, there will be exactly three outcomes

- Win
- Tie
- Lose

With brute force we can permute all possible state and compare

We can use the following data structure: `(suit, value)` where $ suit = ({0,1,2,3}) $ and $ value = ({1...13})$

| Suit | Digit |
| ---- | ----- |
| ♦️   | 0     |
| ♣️   | 1     |
| ♥️   | 2     |
| ♠️   | 3     |

When all 5 river cards are drawn, the only factor affecting the system is distribution and probability

### Logic To Compare Ranks

- From top to bottom check if the 7 cards can form a rank
- Only compare the value when 2 sets have the same rank

## Symbols

♠️♣️♥️♦️🃏0️⃣1️⃣2️⃣3️⃣4️⃣5️⃣6️⃣7️⃣8️⃣9️⃣🔟
