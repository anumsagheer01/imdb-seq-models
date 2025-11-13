# Comparative Analysis of RNN Architectures for Sentiment Classification on IMDB Dataset

## Introduction:
This project is about testing how simple RNN style models behave, using the IMDB movie reviews dataset when I change only one thing at a time. The main things are: model type (RNN / LSTM / BiLSTM), activation (Sigmoid / ReLU / Tanh), optimizer (Adam / SGD / RMSProp), sequence length (25, 50, 100 tokens), and whether I use gradient clipping. Everything else stays fixed so the comparison is fair. I report Accuracy, Macro-F1, and show two plots:
1) Accuracy / F1 vs Sequence Length  
2) Training Loss vs Epochs for the best and the worst configuration.

## Setup
- **Python:** 3.10+  
- **Dependencies:** installed from `requirements.txt` (TensorFlow, NumPy, Pandas, scikit-learn, Matplotlib).  
- **Data expected in `data/`:**  
  - `imdb_25.npz`, `imdb_50.npz`, `imdb_100.npz` (preprocessed arrays)  
  - `word_index.json` (optional tokenizer vocab)  
  If any of these are missing, run the preprocessing notebook first; it saves the files into `data/`.


## Steps to run
1. **Pick the setting I want to test**  
   Choose one sequence length (25, 50, or 100). When I test a different factor, I keep the rest the same.

2. **Build the baseline model**  
   Embedding size = 100, two hidden layers with 64 units, dropout around 0.5, final sigmoid for binary sentiment.  
   I switch only the factor I’m testing (architecture / activation / optimizer / grad clipping / seq length).

3. **Train once, cleanly**  
   - **Epochs:** 10
   - **Batch size:** 32  
   - **Validation split:** 0.1  
   - I don’t touch other hyperparameters while I compare one factor.

4. **Evaluate and log the run**  
   - Compute Accuracy and Macro-F1 on the test set.  
   - Append one row to `results/metrics.csv` with:  
     `Model, Activation, Optimizer, SeqLen, GradClipping, Accuracy, F1, EpochTime(s)`

5. **Make the two plots**  
   - **Accuracy / F1 vs Sequence Length:** compare 25 vs 50 vs 100 using `results/metrics.csv`.  
   - **Training Loss vs Epochs (10 epochs):** plot both train and val loss for the **best** and the **worst** configuration.

6. **Organize outputs**  
   - Tables go in `results/metrics.csv`.  
   - Figures go in `results/plots/`.
   - My write-up is `report.pdf`.

7. **Fair-comparison rule**  
   Change one thing at a time. If I swap optimizer, I keep the same model, same sequence length, same epochs, etc.


## Expected Runtime & Outputs
- **Runtime (CPU only):** a single 10-epoch LSTM (64 units) run usually takes a few minutes; BiLSTM is slower. With a GPU, it’s much faster.  
- **Files/folders:**
  - `data/`: preprocessed `.npz` files (25/50/100)  
  - `results/metrics.csv`: one row per experiment  
  - `results/plots/accuracy_f1_vs_seq_length.png`: comparison across 25/50/100  
  - `results/plots/training_loss_vs_epochs_best.png` and `training_loss_vs_epochs_worst.png`: 10-epoch curves  
  - `report.pdf`: My report with all the analysis
