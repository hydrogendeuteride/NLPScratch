# Natural Language Processing From Scratch

This repository contains implementations of various models for 
Part-of-Speech (POS) tagging, as part of a school NLP assignment. 
The models implemented are:

- Hidden Markov Model (HMM)
- Recurrent Neural Network (RNN)
- Long Short-Term Memory (LSTM)
- Transformer (next word prediction)

## Contents
- [Project structure](#project-structure-)
- [Models Implemented](#models-implemented)
  - [HMM](#HMM)
  - [RNN](#RNN)
  - [LSTM](#LSTM)
  - [SkipGram](#SkipGram)
  - [Transformer](#Transformer)
- [Performance](#performance)
- [Example Dataset](#example-dataset)

## Project Structure 
- **dataset**
  - `tagged_train_mini.txt`
- **HMM**
  - `hmm.py`
  - `test.py`
  - `match.py`
  - `unsupervised.py`
- **RNN**
  - `RNN.py`
  - `RNN_train.py`
  - `RNN_eval.py`
  - `RNN_test.py`
- **LSTM**
  - `LSTM.py`
  - `LSTM_train.py`
  - `LSTM_test.py`
  - `LSTM_eval.py`
- **SkipGram**
  - `word2vec.py`
  - `word2vec_train.py`
  - `torch_skipgram.py`
- **Transformer**
  - `transformer.py`
  - `transformer_train.py`
- **utils**
  - `functions.py`
  - `train.py`
  - `utils.py`

## How To Run (no import errors)
- Always run from the repo root and use module mode so imports resolve:
  - SkipGram training (PyTorch):
    - `python -m SkipGram.torch_skipgram`
  - Transformer training (NumPy/CuPy):
    - `python -m Transformer.transformer_train`

- Alternatively, run the scripts directly from anywhere by setting `PYTHONPATH` to the repo root:
  - `PYTHONPATH=. python SkipGram/torch_skipgram.py`
  - `PYTHONPATH=. python Transformer/transformer_train.py`

The training scripts were updated to resolve dataset/weight paths relative to the repository root and to add the root to `sys.path` automatically. This avoids `ModuleNotFoundError: No module named 'utils'` when not launched from a specific directory.

## Models Implemented
### HMM
Hidden Markov Model POS tagging:
 - **Viterbi tagging**: Implements the Viterbi algorithm for POS tagging (`HMM/hmm.py`)
 - **Evaluation**: Evaluates the performance of the HMM model (`HMM/test.py`)
 - **Baum-Welch algorithm**: Implements the Baum-Welch algorithm for unsupervised training (`HMM/unsupervised.py`)
 - **Baum-Welch evaluation**: Evaluates the Baum-Welch algorithm (incomplete) (`HMM/match.py`)

### RNN
Recurrent Neural Network for POS tagging using only numpy and cupy:
- **RNN implementation**: Core RNN implementation (`RNN/RNN.py`)
- **Training**: Script for training the RNN model (`RNN/RNN_train.py`)
- **Testing**: Script for testing the RNN model (`RNN/RNN_test.py`)
- **Evaluation**: Script for evaluating the RNN model (`RNN/RNN_eval.py`)

### LSTM
Long Short Term Memory network for POS tagging using only numpy and cupy:
- **LSTM implementation**: Core LSTM implementation (`LSTM/LSTM.py`)
- **Training**: Script for training the LSTM model (`LSTM/LSTM_train.py`)
- **Testing**: Script for testing the LSTM model (`LSTM/LSTM_test.py`)
- **Evaluation**: Script for evaluating the LSTM model (`LSTM/LSTM_eval.py`)

### SkipGram
Word2vec implementation for training word embeddings used in RNN, LSTM, and Transformer models:
- **Word2vec numpy implementation**: Core Word2vec implementation using numpy (`SkipGram/word2vec.py`)
- **Word2vec training**: Script for training Word2vec model (`SkipGram/word2vec_train.py`)
- **Pytorch fast word2vec**: Faster Word2vec implementation using PyTorch (`SkipGram/torch_skipgram.py`)

### Transformer
Transformer model for predicting the next word in a sentence using only numpy and cupy:
- **Transformer implementation**: Core Transformer implementation (`Transformer/transformer.py`)
- **Transformer training**: Script for training the Transformer model (`Transformer/transformer_train.py`)

## Performance


## Example Dataset
``` tagged_train.txt
42c027e4ff9730fbb3de84c1af0d2c506e41c3e4::0	LONDON/NNP ,/, England/NNP (/( Reuters/NNP )/) --/: Harry/NNP Potter/NNP star/NN Daniel/NNP Radcliffe/NNP gains/NNS access/NN to/TO a/DT reported/VBN £20/CD million/CD (/( $/$ 41.1/CD million/CD )/) fortune/NN as/IN he/PRP turns/VBZ 18/CD on/IN Monday/NNP ,/, but/CC he/PRP insists/VBZ the/DT money/NN wo/MD n't/RB cast/VB a/DT spell/NN on/IN him/PRP ./.
42c027e4ff9730fbb3de84c1af0d2c506e41c3e4::1	Daniel/NNP Radcliffe/NNP as/IN Harry/NNP Potter/NNP in/IN ``/`` Harry/NNP Potter/NNP and/CC the/DT Order/NN of/IN the/DT Phoenix/NNP ''/'' To/TO the/DT disappointment/NN of/IN gossip/NN columnists/NNS around/IN the/DT world/NN ,/, the/DT young/JJ actor/NN says/VBZ he/PRP has/VBZ no/DT plans/NNS to/TO fritter/VB his/PRP$ cash/NN away/RB on/IN fast/JJ cars/NNS ,/, drink/NN and/CC celebrity/NN parties/NNS ./.
42c027e4ff9730fbb3de84c1af0d2c506e41c3e4::2	``/`` I/PRP do/VBP n't/RB plan/VB to/TO be/VB one/CD of/IN those/DT people/NNS who/WP ,/, as/RB soon/RB as/IN they/PRP turn/VBP 18/CD ,/, suddenly/RB buy/VBP themselves/PRP a/DT massive/JJ sports/NNS car/NN collection/NN or/CC something/NN similar/JJ ,/, ''/'' he/PRP told/VBD an/DT Australian/JJ interviewer/NN earlier/RBR this/DT month/NN ./.
42c027e4ff9730fbb3de84c1af0d2c506e41c3e4::3	``/`` I/PRP do/VBP n't/RB think/VB I/PRP 'll/MD be/VB particularly/RB extravagant/JJ ./.
42c027e4ff9730fbb3de84c1af0d2c506e41c3e4::4	``/`` The/DT things/NNS I/PRP like/IN buying/VBG are/VBP things/NNS that/IN cost/NN about/IN 10/CD pounds/NNS --/: books/NNS and/CC CDs/NNS and/CC DVDs/NNP ./. ''/''
42c027e4ff9730fbb3de84c1af0d2c506e41c3e4::5	At/IN 18/CD ,/, Radcliffe/NNP will/MD be/VB able/JJ to/TO gamble/VB in/IN a/DT casino/NN ,/, buy/VB a/DT drink/NN in/IN a/DT pub/NN or/CC see/VB the/DT horror/NN film/NN ``/`` Hostel/NN :/: Part/NNP II/NNP ,/, ''/'' currently/RB six/CD places/NNS below/IN his/PRP$ number/NN one/CD movie/NN on/IN the/DT UK/NNP box/NN office/NN chart/NN ./.
42c027e4ff9730fbb3de84c1af0d2c506e41c3e4::6	Details/NNS of/IN how/WRB he/PRP 'll/MD mark/VB his/PRP$ landmark/NN birthday/NN are/VBP under/IN wraps/NNS ./.
42c027e4ff9730fbb3de84c1af0d2c506e41c3e4::7	His/PRP$ agent/NN and/CC publicist/NN had/VBD no/DT comment/NN on/IN his/PRP$ plans/NNS ./.
42c027e4ff9730fbb3de84c1af0d2c506e41c3e4::8	``/`` I/PRP 'll/MD definitely/RB have/VB some/DT sort/NN of/IN party/NN ,/, ''/'' he/PRP said/VBD in/IN an/DT interview/NN ./.
42c027e4ff9730fbb3de84c1af0d2c506e41c3e4::9	``/`` Hopefully/NNP none/NN of/IN you/PRP will/MD be/VB reading/VBG about/IN it/PRP ./. ''/''
42c027e4ff9730fbb3de84c1af0d2c506e41c3e4::10	Radcliffe/NNP 's/POS earnings/NNS from/IN the/DT first/JJ five/CD Potter/NNP films/NNS have/VBP been/VBN held/VBN in/IN a/DT trust/NN fund/NN which/WDT he/PRP has/VBZ not/RB been/VBN able/JJ to/TO touch/VB ./.
42c027e4ff9730fbb3de84c1af0d2c506e41c3e4::11	Despite/IN his/PRP$ growing/VBG fame/NN and/CC riches/NNS ,/, the/DT actor/NN says/VBZ he/PRP is/VBZ keeping/VBG his/PRP$ feet/NNS firmly/RB on/IN the/DT ground/NN ./.
42c027e4ff9730fbb3de84c1af0d2c506e41c3e4::12	``/`` People/NNS are/VBP always/RB looking/VBG to/TO say/VB 'kid/CD star/NN goes/VBZ off/IN the/DT rails/NNS ,/, '/'' ''/'' he/PRP told/VBD reporters/NNS last/JJ month/NN ./.
42c027e4ff9730fbb3de84c1af0d2c506e41c3e4::13	``/`` But/CC I/PRP try/VBP very/RB hard/RB not/RB to/TO go/VB that/DT way/NN because/IN it/PRP would/MD be/VB too/RB easy/JJ for/IN them/PRP ./. ''/''
42c027e4ff9730fbb3de84c1af0d2c506e41c3e4::14	His/PRP$ latest/JJS outing/NN as/IN the/DT boy/NN wizard/NN in/IN ``/`` Harry/NNP Potter/NNP and/CC the/DT Order/NN of/IN the/DT Phoenix/NNP ''/'' is/VBZ breaking/VBG records/NNS on/IN both/DT sides/NNS of/IN the/DT Atlantic/NNP and/CC he/PRP will/MD reprise/VB the/DT role/NN in/IN the/DT last/JJ two/CD films/NNS ./.
42c027e4ff9730fbb3de84c1af0d2c506e41c3e4::15	Watch/VB I-Reporter/NNP give/VB her/PRP$ review/NN of/IN Potter/NNP 's/POS latest/JJS »/NN ./.
42c027e4ff9730fbb3de84c1af0d2c506e41c3e4::16	There/EX is/VBZ life/NN beyond/IN Potter/NNP ,/, however/RB ./.

```
```raw_train.txt
42c027e4ff9730fbb3de84c1af0d2c506e41c3e4::1	Daniel Radcliffe as Harry Potter in "Harry Potter and the Order of the Phoenix" To the disappointment of gossip columnists around the world, the young actor says he has no plans to fritter his cash away on fast cars, drink and celebrity parties.
42c027e4ff9730fbb3de84c1af0d2c506e41c3e4::2	"I don't plan to be one of those people who, as soon as they turn 18, suddenly buy themselves a massive sports car collection or something similar," he told an Australian interviewer earlier this month.
42c027e4ff9730fbb3de84c1af0d2c506e41c3e4::3	"I don't think I'll be particularly extravagant.
42c027e4ff9730fbb3de84c1af0d2c506e41c3e4::4	"The things I like buying are things that cost about 10 pounds -- books and CDs and DVDs."
42c027e4ff9730fbb3de84c1af0d2c506e41c3e4::5	At 18, Radcliffe will be able to gamble in a casino, buy a drink in a pub or see the horror film "Hostel: Part II," currently six places below his number one movie on the UK box office chart.
42c027e4ff9730fbb3de84c1af0d2c506e41c3e4::6	Details of how he'll mark his landmark birthday are under wraps.
42c027e4ff9730fbb3de84c1af0d2c506e41c3e4::7	His agent and publicist had no comment on his plans.
42c027e4ff9730fbb3de84c1af0d2c506e41c3e4::8	"I'll definitely have some sort of party," he said in an interview.
42c027e4ff9730fbb3de84c1af0d2c506e41c3e4::9	"Hopefully none of you will be reading about it."
42c027e4ff9730fbb3de84c1af0d2c506e41c3e4::10	Radcliffe's earnings from the first five Potter films have been held in a trust fund which he has not been able to touch.
42c027e4ff9730fbb3de84c1af0d2c506e41c3e4::11	Despite his growing fame and riches, the actor says he is keeping his feet firmly on the ground.
42c027e4ff9730fbb3de84c1af0d2c506e41c3e4::12	"People are always looking to say 'kid star goes off the rails,'" he told reporters last month.
42c027e4ff9730fbb3de84c1af0d2c506e41c3e4::13	"But I try very hard not to go that way because it would be too easy for them."
42c027e4ff9730fbb3de84c1af0d2c506e41c3e4::14	His latest outing as the boy wizard in "Harry Potter and the Order of the Phoenix" is breaking records on both sides of the Atlantic and he will reprise the role in the last two films.
42c027e4ff9730fbb3de84c1af0d2c506e41c3e4::15	Watch I-Reporter give her review of Potter's latest » .
```
