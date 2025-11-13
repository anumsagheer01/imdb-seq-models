import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def basic_clean(s):
    s = s.lower().replace("<br />"," ")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def build_tokenizer(texts, vocab=10000):
    tok = Tokenizer(num_words=vocab, oov_token="<OOV>")
    tok.fit_on_texts(texts)
    return tok

def to_fixed_len(seqs, L):
    return pad_sequences(seqs, maxlen=L, padding="post", truncating="post")
