import os
import pickle
import sys

# Make repo root importable regardless of how this file is executed
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Support both package and script execution
try:
    from .transformer import Transformer
except ImportError:
    from transformer import Transformer

from utils.decoding import generate_transformer_ids
from utils.utils import build_vocab, count_word_POS, read_file_to_list, reader


def _decode_word_ids(ids, idx_to_word):
    words = []
    for i in ids:
        w = idx_to_word.get(int(i), "<UNK>")
        if w == "<END>":
            break
        if w in ("<PAD>", "<START>"):
            continue
        words.append(w)
    return " ".join(words)


def _decode_bpe_ids(ids, idx_to_token):
    toks = []
    for i in ids:
        t = idx_to_token.get(int(i), "<UNK>")
        if t == "<EOS>":
            break
        if t in ("<PAD>", "<BOS>"):
            continue
        toks.append(t)
    return " ".join(toks).replace("@@ ", "")


def _load_word_vocab_from_pickle(vocab_path):
    with open(vocab_path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and "word_to_idx" in data:
        word_to_idx = data["word_to_idx"]
    elif isinstance(data, dict):
        word_to_idx = data
    else:
        raise ValueError("Unsupported vocab pickle format (expected dict).")

    if word_to_idx.get("<PAD>") != 0:
        raise ValueError(f"Expected '<PAD>' id to be 0, got {word_to_idx.get('<PAD>')}.")
    return word_to_idx


def _build_word_vocab_from_dataset(dataset_path):
    data_line = read_file_to_list(dataset_path)
    processed = reader(data_line)
    pos_cnt, word_cnt = count_word_POS(processed)
    word_to_idx, _tag_to_idx = build_vocab(word_cnt, pos_cnt)
    return word_to_idx


## Configuration — edit these variables directly ##
MODE = "bpe"                                                   # "word" or "bpe"
MODEL_PATH = os.path.join(REPO_ROOT, "transformer_model.pth")
PROMPT = "the"                                                  # prompt text (whitespace tokenized)
MAX_NEW_TOKENS = 30
TOPK = 5
TOPP = 0.9
TEMPERATURE = 1.0
SEED = None
USE_GPU = False
NUM_HEADS = 8                                                   # needed for legacy checkpoints without config
MAX_LEN = 320                                                   # context length (positional encoding size)

# word-mode only
DATASET = os.path.join(REPO_ROOT, "dataset", "tagged_train.txt")
VOCAB = None                                                    # optional vocab pickle path

# bpe-mode only
BPE_MODEL = os.path.join(REPO_ROOT, "weight", "bpe.pt")


def main():
    prompt = (PROMPT or "").strip()
    tokens = prompt.split() if prompt else []

    model = Transformer.load(
        MODEL_PATH,
        num_heads=NUM_HEADS,
        max_len=MAX_LEN,
        use_gpu=USE_GPU,
    )
    model.eval()

    if MODE == "bpe":
        try:
            from .bpe_encoder import BPEEncoder
        except ImportError:
            try:
                from bpe_encoder import BPEEncoder
            except ImportError as e:
                raise SystemExit("BPE mode requires torch (failed to import Transformer.bpe_encoder).") from e
        bpe = BPEEncoder.load(BPE_MODEL)
        if int(getattr(model, "vocab_size", 0)) != int(getattr(bpe, "vocab_size", 0)):
            raise SystemExit(
                f"Vocab size mismatch: model.vocab_size={model.vocab_size}, bpe.vocab_size={bpe.vocab_size}."
            )
        prompt_ids = bpe.encode_ids(tokens, add_bos=True, add_eos=False)
        stop_id = bpe.token_to_id.get("<EOS>", 3)
        out_ids = generate_transformer_ids(
            model,
            prompt_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            topk=TOPK,
            topp=TOPP,
            temperature=TEMPERATURE,
            stop_id=stop_id,
            pad_id=bpe.pad_id,
            seed=SEED,
        )
        idx_to_token = {int(i): t for i, t in bpe.id_to_token.items()}
        print(_decode_bpe_ids(out_ids, idx_to_token))
        return

    # word mode
    if VOCAB:
        word_to_idx = _load_word_vocab_from_pickle(VOCAB)
    else:
        word_to_idx = _build_word_vocab_from_dataset(DATASET)

    if int(word_to_idx.get("<PAD>", -1)) != 0 or int(word_to_idx.get("<UNKNOWN>", -1)) != 1:
        raise SystemExit(
            f"Unexpected special ids: <PAD>={word_to_idx.get('<PAD>')}, <UNKNOWN>={word_to_idx.get('<UNKNOWN>')}."
        )
    max_id = max((int(i) for i in word_to_idx.values()), default=0)
    if max_id >= int(getattr(model, "vocab_size", 0)):
        raise SystemExit(
            f"Vocab mismatch: max_id={max_id} >= model.vocab_size={model.vocab_size}. "
            "Build/load the same vocab used during training."
        )

    if tokens and tokens[0] != "<START>":
        tokens = ["<START>"] + tokens
    unk_id = word_to_idx.get("<UNKNOWN>", 1)
    prompt_ids = [int(word_to_idx.get(t, unk_id)) for t in tokens]
    stop_id = word_to_idx.get("<END>")
    out_ids = generate_transformer_ids(
        model,
        prompt_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        topk=TOPK,
        topp=TOPP,
        temperature=TEMPERATURE,
        stop_id=stop_id,
        pad_id=0,
        seed=SEED,
    )
    idx_to_word = {int(i): w for w, i in word_to_idx.items()}
    print(_decode_word_ids(out_ids, idx_to_word))


if __name__ == "__main__":
    main()
