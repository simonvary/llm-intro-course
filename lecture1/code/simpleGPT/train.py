from utils import load_and_preprocess_data, train_step
from model import simpleGPT
import tiktoken
import torch
import numpy as np
from tqdm import tqdm
import os

CHECKPOINT_DIR = "simpleGPT"
CHECKPOINT_EVERY = 400
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

METRICS_PATH = os.path.join(CHECKPOINT_DIR, "simpleGPT-metrics.npz")

def save_checkpoint(step, model, optimizer, losses, perplexities):
    path = os.path.join(CHECKPOINT_DIR, "simpleGPT-ckpt_latest.pt")
    torch.save({
        "step":          step,
        "model":         model.state_dict(),
        "optimizer":     optimizer.state_dict(),
        "losses":        losses,
        "perplexities":  perplexities,
    }, path)
    print(f"[checkpoint] saved : {path}")

def save_metrics(losses, perplexities):
    np.savez(METRICS_PATH, losses=np.array(losses), perplexities=np.array(perplexities))
    print(f"[metrics] saved → {METRICS_PATH}")

tokenizer = tiktoken.get_encoding("gpt2")

vocab_size = tokenizer.n_vocab
num_transformer_blocks = 4
maxlen = 256
embed_dim = 256
num_heads = 8
feed_forward_dim = 256
batch_size = 72
num_epochs = 1
top_k = 10

device = "cuda" #torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_path = 'TinyStories-train.txt'

text_dl = load_and_preprocess_data(data_path, batch_size, maxlen, num_epochs)

model = simpleGPT(
    maxlen=maxlen,
    vocab_size=vocab_size,
    embed_dim=embed_dim,
    num_heads=num_heads,
    feed_forward_dim=feed_forward_dim,
    num_transformer_blocks=num_transformer_blocks,
    tokenizer=tokenizer,
    top_k=top_k
).to(device)

optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)

prep_target_batch = torch.vmap(
    lambda tokens: torch.cat((tokens[1:], tokens.new_zeros(1)), dim=0)
)

prompt = 'Once upon a time'
start_tokens = tokenizer.encode(prompt)[:maxlen]
print('Initial Text:')
_ = model.generate_text(maxlen,start_tokens)


losses = []
perplexities = []
for step,batch in tqdm(enumerate(text_dl)):
    input_batch = torch.tensor(np.array(batch)).T.to(device)
    target_batch = prep_target_batch(input_batch).to(device)
    loss = train_step(model,optimizer,input_batch,target_batch)

    loss_val = loss.detach().cpu()
    losses.append(loss_val)
    perplexities.append(float(torch.exp(loss_val)))

    if (step + 1) % CHECKPOINT_EVERY == 0:
        save_checkpoint(step + 1, model, optimizer, losses, perplexities)
        save_metrics(losses, perplexities)
        print(f"Step: {step}, Avg Loss: {np.mean(losses[-200:])} and Loss: {loss}")
        print("Generating Text:")
        _ = model.generate_text(maxlen,start_tokens)
        print(f"Perplexity: {np.mean(perplexities[-200:]):.2f}")

save_metrics(losses, perplexities)

    