from sklearn.metrics import f1_score
from tensorflow.keras import optimizers

def make_optimizer(name="adam", lr=1e-3, clip=False):
    clipnorm = 1.0 if clip else None
    n = name.lower()
    if n == "adam":    return optimizers.Adam(learning_rate=lr, clipnorm=clipnorm)
    if n == "sgd":     return optimizers.SGD(learning_rate=lr, clipnorm=clipnorm)
    if n == "rmsprop": return optimizers.RMSprop(learning_rate=lr, clipnorm=clipnorm)
    raise ValueError("optimizer must be adam/sgd/rmsprop")

def train_and_eval(model, X_train, y_train, X_test, y_test,
                   opt="adam", lr=1e-3, clip=False, epochs=10, batch=32, val_split=0.1):
    model.compile(loss="binary_crossentropy",
                  optimizer=make_optimizer(opt, lr, clip),
                  metrics=["accuracy"])
    history = model.fit(X_train, y_train,
                        validation_split=val_split,
                        epochs=min(epochs,10),
                        batch_size=batch,
                        verbose=1)
    probs = model.predict(X_test).ravel()
    preds = (probs >= 0.5).astype("int32")
    acc = float((preds == y_test).mean())
    f1  = float(f1_score(y_test, preds, average="macro"))
    return acc, f1, history.history  # keep history for loss plots
