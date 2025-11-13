from tensorflow.keras import layers, models

def build_model(arch="lstm", seq_len=50, vocab_size=10000, units=64, dropout=0.5, activation="relu"):
    model = models.Sequential()
    model.add(layers.Embedding(input_dim=vocab_size+1, output_dim=100, input_length=seq_len))
    if arch == "rnn":
        model.add(layers.SimpleRNN(units, activation=activation, return_sequences=True))
        model.add(layers.Dropout(dropout))
        model.add(layers.SimpleRNN(units, activation=activation))
    elif arch == "lstm":
        model.add(layers.LSTM(units, activation=activation, return_sequences=True))
        model.add(layers.Dropout(dropout))
        model.add(layers.LSTM(units, activation=activation))
    elif arch == "bilstm":
        model.add(layers.Bidirectional(layers.LSTM(units, activation=activation, return_sequences=True)))
        model.add(layers.Dropout(dropout))
        model.add(layers.Bidirectional(layers.LSTM(units, activation=activation)))
    else:
        raise ValueError("arch must be rnn/lstm/bilstm")
    model.add(layers.Dense(1, activation="sigmoid"))
    return model
