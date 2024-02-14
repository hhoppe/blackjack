# bjnb: Blackjack Notebook

[Hugues Hoppe](https://hhoppe.com/)
&nbsp;&nbsp;&mdash;&nbsp;
&nbsp; [**[Open in Colab]**](https://colab.research.google.com/github/hhoppe/blackjack/blob/main/blackjack.ipynb)
&nbsp; [**[Open in Kaggle]**](https://www.kaggle.com/notebooks/welcome?src=https://github.com/hhoppe/blackjack/blob/main/blackjack.ipynb)
&nbsp; [**[Open in MyBinder]**](https://mybinder.org/v2/gh/hhoppe/blackjack/main?filepath=blackjack.ipynb)
&nbsp; [**[GitHub source]**](https://github.com/hhoppe/blackjack)

Blackjack &mdash; *"the most widely played casino banking game in the world"*.


**Goals**:

- Solution methods for both:

  1. [Probabilistic analysis](https://colab.research.google.com/github/hhoppe/blackjack/blob/main/blackjack.ipynb#Probabilistic-analysis)
     of blackjack actions and outcomes under many strategies, and

  2. [Monte Carlo simulation](https://colab.research.google.com/github/hhoppe/blackjack/blob/main/blackjack.ipynb#Monte-Carlo-simulation)
     for cut-card effects and precise split-hand rewards.

- Support for many [rule variations](https://colab.research.google.com/github/hhoppe/blackjack/blob/main/blackjack.ipynb#Define-Rules)
  (12 parameters including #decks, dealer hit soft17, cut-card, ...)

- Optimal [action tables](https://colab.research.google.com/github/hhoppe/blackjack/blob/main/blackjack.ipynb#Tables-for-basic-strategy) for
  [basic strategy](https://colab.research.google.com/github/hhoppe/blackjack/blob/main/blackjack.ipynb#Define-Actions-and-Strategy)
  under any rules, reproducing
  [Wikipedia](https://en.wikipedia.org/wiki/Blackjack#Basic_strategy) and
  [WizardOfOdds](https://wizardofodds.com/games/blackjack/strategy/calculator/) results.

- Six separate [composition-dependent strategies](https://colab.research.google.com/github/hhoppe/blackjack/blob/main/blackjack.ipynb#Define-Actions-and-Strategy)
  based on different levels of *attention*.
  <!--(initial cards, all hand cards, cards in *prior split hands*, ...).-->

- Computation of
  [house edge](https://colab.research.google.com/github/hhoppe/blackjack/blob/main/blackjack.ipynb#House-edge-results)
  under any rules, with either basic or composition-dependent strategies.

- Comparisons with online
  [hand calculator](https://colab.research.google.com/github/hhoppe/blackjack/blob/main/blackjack.ipynb#Hand-calculator-results) and
  [house edge calculator](https://colab.research.google.com/github/hhoppe/blackjack/blob/main/blackjack.ipynb#House-edge-results) results.

- Novel analysis and visualization of the
  [*cut-card effect*](https://colab.research.google.com/github/hhoppe/blackjack/blob/main/blackjack.ipynb#Effect-of-using-a-cut-card)
  and its [surprising oscillations](https://colab.research.google.com/github/hhoppe/blackjack/blob/main/blackjack.ipynb#cut-card-graph).

- Open-source Python, sped up with jitting (\~30x) and multiprocessing (\~10x),
  simulating ~$10^{8}$ hands/s.


**Versions**:

- 1.0 (March 2022): use probabilistic analysis for basic strategy tables and
  approximate house edge.
- 2.0 (July 2022): add Monte Carlo simulation, hand analysis,
  and cut-card analysis.


**Running this Jupyter notebook**:

- The notebook requires Python 3.10 or later.
- We recommend starting a Jupyter server on a local machine with a fast multi-core CPU. <br/>
  (The notebook can also be executed on a
  [Colab server](
   https://colab.research.google.com/github/hhoppe/blackjack/blob/main/blackjack.ipynb),
  but it runs ~20x slower due to the older, shared processor.)
- Within a Linux environment (e.g.,
  [Windows Subsystem for Linux](https://docs.microsoft.com/en-us/windows/wsl/install)):

```bash
    sudo apt install python3-pip
    python3 -m pip install --upgrade pip
    pip install jupyterlab jupytext matplotlib numba tqdm
    jupyter lab --no-browser
```

- Open the URL (output by `jupyter lab`) using a web browser (e.g., Google Chrome on Windows).
- Load the notebook (`*.ipynb` file).
- Evaluate all cells in `Code library` and then selectively evaluate `Results`.
- Adjust the `EFFORT` global variable to trade off speed and accuracy.


**References**:

- https://en.wikipedia.org/wiki/Blackjack
- https://wizardofodds.com/games/blackjack/basics/#rules
- https://www.casinoguardian.co.uk/blackjack/ -- rich; explore more?
- https://wizardofvegas.com/guides/blackjack-survey/
- https://www.blackjackinfo.com/ -- created by Ken Smith; explore?
- https://www.onlinegambling.com/blackjack/odds/
- https://www.onlineunitedstatescasinos.com/las-vegas/blackjack/
- https://en.wikipedia.org/wiki/Gambling_mathematics
- https://www.gamingtheodds.com/blackjack/house-edge/ -- looks like truncated results of
  WizardOfOdds assuming no late surrender.
- https://github.com/johntelforduk/blackjack
  -- Python simulator to evaluate house edge for various strategies.
- [First reddit post](
 https://www.reddit.com/r/blackjack/comments/t9ygkm/python_notebook_to_analyze_blackjack_optimal/)

(See also references in
 [Tables for basic strategy](https://colab.research.google.com/github/hhoppe/blackjack/blob/main/blackjack.ipynb#Tables-for-basic-strategy),
 [Hand calculators](https://colab.research.google.com/github/hhoppe/blackjack/blob/main/blackjack.ipynb#Hand-calculators), and
 [House edge calculators](https://colab.research.google.com/github/hhoppe/blackjack/blob/main/blackjack.ipynb#House-edge-calculators).)
