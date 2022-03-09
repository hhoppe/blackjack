# Blackjack game probabilistic analysis

[Hugues Hoppe](https://hhoppe.com/) &emsp; first version in March 2022
&emsp;
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hhoppe/blackjack/blob/main/blackjack.ipynb)

(I was taking a reinforcement learning course which considered a simplified version of the blackjack game &mdash; *&ldquo;the most widely played casino banking game in the world&rdquo;* &mdash; this became a side project.)

**Goals**:

- Probabilistic analysis of
[blackjack](https://en.wikipedia.org/wiki/Blackjack) actions and outcomes
(without Monte Carlo simulation).

- Support for many rule variations, including:
  - Number of decks (1, 2, 4, 6, 8, infinite shoe)
  - Blackjack payout (e.g. 3:2, 6:5, 1:1)
  - Dealer stand/hit on soft total of 17
  - Loss of original bet or all bets to a dealer's blackjack.
  - Double on any two cards or with mininum total
  - Double after splitting (either any card or non-ace)
  - Resplit to any number of hands
  - Repslit aces

- Optimal-action tables for [basic strategy](https://en.wikipedia.org/wiki/Blackjack#Basic_strategy) under any rules.

- Composition-dependent strategy (considering not just player total but individual card values).

- Computation of house edge under any rules, with either basic or composition-dependent strategy.

- Reproduction of the
[basic strategy tables](https://en.wikipedia.org/wiki/Blackjack#Basic_strategy) and
[house edge results](https://en.wikipedia.org/wiki/Blackjack#Rule_variations_and_effects_on_house_edge)
listed in online sources.

**References**:
- https://en.wikipedia.org/wiki/Blackjack
- https://wizardofvegas.com/guides/blackjack-survey/
- https://www.onlinegambling.com/blackjack/odds/
- https://www.onlineunitedstatescasinos.com/las-vegas/blackjack/
- https://vegasadvantage.com/best-las-vegas-blackjack-games/
