import torch
from torch import Tensor, nn, optim
from torch.utils.data import Dataset, DataLoader, IterableDataset
from transformers import get_linear_schedule_with_warmup, PreTrainedTokenizerFast
from tqdm import tqdm
from dataclasses import dataclass
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast
import time
from torch.optim import AdamW
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')
try:
    import torch._dynamo as dynamo
    dynamo.config.suppress_errors = True
    dynamo.config.verbose = False
except ImportError:
    pass
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
@dataclass
class Config:
    vocab_size: int
    seq_len: int
    d_model: int
    num_heads: int
    num_layers: int
    d_ff: int
    dropout: float = 0.1
    grad_clip_norm: float = 1.0
    lr: float = 6e-4
    batch_size: int = 64
    epochs: int = 2
    steps_per_epoch: int = 28000
    report_interval: int = 1000000  
    betas: tuple = (0.9, 0.95)
    weight_decay: float = 0.01
    use_fused: bool = True
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
class TinyStoriesDataset(IterableDataset, Dataset):
    def __init__(
        self,
        token_file_path: str = None,
        seq_len: int = 128,
        is_streaming: bool = True,
        device: str = 'cuda',
        prefetch_size: int = 4096,
        token_ids: torch.Tensor = None
    ):
        super().__init__()
        self.seq_len = seq_len
        self.is_streaming = is_streaming
        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        self.prefetch_size = prefetch_size

        if token_ids is not None:
            self.token_ids = token_ids  # Keep on CPU initially
        elif token_file_path is not None:
            self.token_ids = torch.load(token_file_path, map_location='cpu')
        else:
            raise ValueError("Either 'token_file_path' or 'token_ids' must be provided.")

        self.token_len = len(self.token_ids)

        if self.token_len < self.seq_len + 1:
            raise ValueError(f"Token length ({self.token_len}) is too short for seq_len ({self.seq_len}).")

    def __len__(self):
        if self.is_streaming:
            raise NotImplementedError("Length is not defined for streaming dataset.")
        return self.token_len - self.seq_len

    def __getitem__(self, idx):
        x = self.token_ids[idx:idx + self.seq_len].to(self.device, non_blocking=True)
        y = self.token_ids[idx + 1:idx + self.seq_len + 1].to(self.device, non_blocking=True)
        return x, y

    def __iter__(self):
        if not self.is_streaming:
            # Support iteration for non-streaming mode
            for idx in range(len(self)):
                yield self.__getitem__(idx)
        else:
            while True:
                idxs = torch.randint(
                    0, self.token_len - self.seq_len - 1,
                    (self.prefetch_size,), device='cpu'
                )
                for idx in idxs:
                    x = self.token_ids[idx:idx + self.seq_len].to(self.device, non_blocking=True)
                    y = self.token_ids[idx + 1:idx + self.seq_len + 1].to(self.device, non_blocking=True)
                    yield x, y
class GPT2Block(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask: torch.Tensor):
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=mask)
        x = x + attn_out
        h = self.ln2(x)
        return x + self.ff(h)

class GPT2Simple(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([
            GPT2Block(cfg.d_model, cfg.num_heads, cfg.d_ff, cfg.dropout)
            for _ in range(cfg.num_layers)
        ])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        bool_mask = torch.triu(torch.ones(cfg.seq_len, cfg.seq_len, dtype=torch.bool), diagonal=1)
        self.register_buffer('causal_mask', bool_mask)

    def forward(self, input_ids: Tensor):
        bsz, seqlen = input_ids.size()
        x = self.tok_emb(input_ids) + self.pos_emb(
            torch.arange(seqlen, device=input_ids.device).unsqueeze(0).expand(bsz, -1)
        )
        mask = self.causal_mask[:seqlen, :seqlen]
        for blk in self.blocks:
            x = blk(x, mask)
        return self.head(self.ln_f(x))



def train_epoch(model, loader, optimizer, scheduler, device, cfg, scaler):
    model.train()
    total_loss = 0.0
    total_token=0.0
    tokens_processed = 0
    loss_accum = 0.0
    accum_count = 0
    loss_per_interval = []
    it = iter(loader)
    
    progress_bar = tqdm(range(cfg.steps_per_epoch), desc="Training", dynamic_ncols=True)
    
    for step in progress_bar:
        x, y = next(it)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        with autocast():
            logits = model(x)
            loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), y.view(-1))
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        loss_item = loss.item()
        total_loss += loss_item
        loss_accum += loss_item
        accum_count += 1
        tokens_processed += cfg.batch_size * cfg.seq_len
        total_token += cfg.batch_size * cfg.seq_len
        
        if tokens_processed >= cfg.report_interval:
            avg_loss = loss_accum / accum_count
            loss_per_interval.append(avg_loss)
            tr_ppl = torch.exp(torch.tensor(avg_loss).clamp(max=100))
            progress_bar.set_postfix(loss=avg_loss, ppl=tr_ppl.item())
            loss_accum = 0.0
            accum_count = 0
            tokens_processed = 0
    
    if accum_count > 0:
        print(f"At the end of the epoch, after processing {tokens_processed:,} tokens in the last interval")

        
    print(f"In the end of the epoch, total processed {total_token:,} tokens")
    print(f"===============================================================")
    return loss_per_interval

def eval_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x, y in tqdm(loader, desc='Eval'):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            total_loss += nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), y.view(-1)).item()
    avg_loss = total_loss / len(loader)
    return avg_loss, torch.exp(torch.tensor(avg_loss)).item()

def generate_text(model, tokenizer, prompt: str, max_new: int, device: str, seq_len: int, top_k: int = 50, temperature: float = 1.0) -> str:
    model.eval()
    with torch.no_grad():
        tokens = tokenizer.encode(prompt, return_tensors='pt').to(device)
        prompt_len = tokens.size(1)
        
        if prompt_len > seq_len:
            tokens = tokens[:, -seq_len:]
            prompt_len = seq_len

        seq = tokens
        for _ in range(max_new):
            context = seq[:, -seq_len:]
            logits = model(context)
            logits = logits[:, -1, :]
            logits = logits / temperature
            top_k_logits, top_k_indices = torch.topk(logits, top_k)
            probs = torch.softmax(top_k_logits, dim=-1)
            next_tok_idx = torch.multinomial(probs, num_samples=1)
            next_tok = top_k_indices.gather(-1, next_tok_idx)
            seq = torch.cat([seq, next_tok], dim=1)

        generated_text = tokenizer.decode(seq[0].tolist())
    return generated_text



def main():
    scaler = GradScaler()
    cfg = Config(10000, 128, 768, 8, 2, 3072, batch_size=64, epochs=2, steps_per_epoch=28000, report_interval=10**6)  # 1 million tokens
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    if device.type == 'cuda':
        print('GPU:', torch.cuda.get_device_name(0))

    tokenizer = PreTrainedTokenizerFast(tokenizer_file='bpe-tokenizer_tinystories.json', pad_token='<|pad|>')
    valid_ids = torch.load('tokenized-valid-samples_vocab-10k.pt', map_location='cpu')

    train_loader = DataLoader(
        TinyStoriesDataset(
            token_file_path='tokenized-train-samples_vocab-10k.pt',
            seq_len=cfg.seq_len,
            is_streaming=True,
            device='cuda',
            prefetch_size=cfg.batch_size * 2
        ),
        batch_size=cfg.batch_size,
        pin_memory=False,
        drop_last=True,
        num_workers=0
    )

    valid_loader = DataLoader(
        TinyStoriesDataset(
            token_ids=valid_ids,
            seq_len=cfg.seq_len,
            is_streaming=False,
            device='cuda'
        ),
        batch_size=cfg.batch_size,
        pin_memory=False,
        num_workers=0
    )
    
    model = GPT2Simple(cfg).to(device)
    print(f"Total number of parameters: {count_parameters(model):,}")
    optimizer = AdamW(model.parameters(), lr=cfg.lr,betas=cfg.betas,
    weight_decay=cfg.weight_decay,
    fused=cfg.use_fused)
    total_steps = cfg.epochs * cfg.steps_per_epoch
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    history = {'tokens': [], 'train_loss': [], 'valid_loss': []}

    for epoch in range(1, cfg.epochs + 1):
        print(f'=== Epoch {epoch}/{cfg.epochs} ===')
        
        start_time = time.time()  # Start time for epoch

        loss_per_interval= train_epoch(model, train_loader, optimizer, scheduler, device, cfg, scaler)

        prompts = ["Hi Jane, have you seen Alice? I can’t find her anywhere”, said Jack.",
                  "Max had two dogs. One was white and the other was black. Max walked up the street and saw a kid with a dog. He told the kid, ”I see you have a Brown dog. I also have",
                  "Anne had a piece of candy in her left pocket and a piece of chocolate in her right pocket. Anne’s mom asked her, ”Anne, what is that you have in your left pocket?” "]
        
        for prompt in prompts:
            sample = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new=64,
                device=device,
                seq_len=cfg.seq_len,
                temperature=0.8
            )
            print(f"sample:: {sample}")
            print(f"===============================================================")


        val_loss, val_ppl = eval_epoch(model, valid_loader, device)
        epoch_duration = time.time() - start_time  # End time for epoch
        epoch_duration_str = time.strftime("%H:%M:%S", time.gmtime(epoch_duration))  # Format duration
        print(f'Epoch {epoch} completed in {epoch_duration_str}.')

        
        
        for i in range(len(loss_per_interval)):
            total = history['tokens'][-1] + cfg.report_interval if history['tokens'] else cfg.report_interval
            history['tokens'].append(total)
            history['train_loss'].append(loss_per_interval[i])


        history['valid_loss'].append(val_loss)
        
        
        print(f'Valid: loss={val_loss:.4f}, ppl={val_ppl:.2f}')
        
        torch.save(model.state_dict(), f'ckpt_epoch{epoch}.pt')

    # Plotting the loss curve
    
    plt.figure(figsize=(10, 6))  
    plt.plot(
        history['tokens'],
        history['train_loss'],
        label='Train Loss',
        color='royalblue',
        linewidth=2.0,
        marker='o',
        markersize=5
    )

    # Optional
    #for x, y in zip(history['tokens'], history['train_loss']):
       # plt.text(x, y, f'{y:.2f}', fontsize=8, ha='right', va='bottom', color='black')

    plt.xlabel('Tokens Processed', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss vs. Tokens Processed', fontsize=14, weight='bold')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
