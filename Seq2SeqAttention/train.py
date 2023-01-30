import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
import numpy as np
import spacy
import random
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import en_core_web_sm
import de_core_news_md

from utils import translate_sentence, bleu

tokenizer_eng = en_core_web_sm.load()
tokenizer_ger = de_core_news_md.load()


def get_tokenized_ger(text):
    return [token.text for token in tokenizer_ger(text)]


def get_tokenized_eng(text):
    return [token.text for token in tokenizer_eng(text)]


englishField = Field(
    tokenize=get_tokenized_eng, lower=True, init_token="<sos>", eos_token="<eos>"
)
germanField = Field(
    tokenize=get_tokenized_ger, lower=True, init_token="<sos>", eos_token="<eos>"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, iterator, criterion):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for batch in iterator:
            src_data = batch.src.to(device)
            trg_data = batch.trg.to(device)

            output = model(src_data, trg_data)

            output = output[1:].reshape(-1, output.shape[2])
            target = trg_data[1:].reshape(-1)

            loss = criterion(output, target)
            epoch_loss += loss.item()

    model.train()

    return epoch_loss / len(iterator)


class Encoder(nn.Module):
    def __init__(self, input_size, emb_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(input_size, emb_size)
        self.lstm = nn.LSTM(
            emb_size, hidden_size, num_layers, dropout=dropout, bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x):
        # x shape: (seq_len, batch_size)

        embedding = self.dropout(self.embedding(x))
        # embedding shape: (seq_len, batch_size, emb_size)

        outputs, (hidden, cell) = self.lstm(embedding)
        # outputs shape: (seq_len, batch_size, hidden_size)

        hidden = self.fc(torch.cat((hidden[0:1], hidden[1:2]), dim=2))
        cell = self.fc(torch.cat((cell[0:1], cell[1:2]), dim=2))

        return outputs, hidden, cell


class Attention(nn.Module):
    def __init__(self, encoder_hidden_size, decoder_hidden_size):
        super().__init__()

        self.attn = nn.Linear(
            encoder_hidden_size * 2 + decoder_hidden_size, decoder_hidden_size
        )
        self.v = nn.Linear(decoder_hidden_size, 1, bias=False)

    def forward(self, hidden, enc_outputs):
        # hidden shape: (batch_size, dec_hidden_size)
        # enc_outputs: (src_len, batch_size, 2 * enc_hidden_size)

        src_len = enc_outputs.shape[0]

        # repeat decoder hidden state src_len times
        hidden = hidden.repeat(src_len, 1, 1)
        # print(hidden.shape)
        hidden = hidden.permute(1, 0, 2)

        enc_outputs = enc_outputs.permute(1, 0, 2)
        # shape : (batch_size, src_len, 2 * enc_hidden_size)

        energy = F.softmax(self.attn(torch.cat((hidden, enc_outputs), dim=2)))
        # energy shape: (batch_size, src_len, decoder_hidden_size)

        attention = self.v(energy).squeeze(2)

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(
        self,
        input_size,
        emb_size,
        hidden_size,
        output_size,
        num_layers,
        dropout,
        attention,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.attention = attention
        self.embedding = nn.Embedding(input_size, emb_size)
        self.lstm = nn.LSTM(hidden_size * 2 + emb_size, hidden_size, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell, encoder_outputs):
        # x shape: (batch_size)
        x = x.unsqueeze(0)

        embedding = self.dropout(self.embedding(x))
        # embdding shape: (1, batch_size, emb_size)

        a = self.attention(hidden, encoder_outputs)
        # a shape: (batch_size, src_len)

        a = a.unsqueeze(1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        weighted = torch.bmm(a, encoder_outputs)
        # weighted shape: (batch_size, 1, hidden_size*2)

        weighted = weighted.permute(1, 0, 2)

        lstm_input = torch.cat((embedding, weighted), dim=2)
        # lstm_input shape: (1, batch_size, emb_size + 2*hidden_size)

        outputs, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        # outputs shape: (1, batch_size, hidden_size)

        preds = self.fc(outputs)

        return preds.squeeze(0), hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_force_ratio=0.5):
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = len(englishField.vocab)

        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)

        encoder_outputs, hidden, cell = self.encoder(source)

        # Grab the first input to the Decoder which will be <SOS> token
        x = target[0]

        for word in range(1, target_len):
            output, hidden, cell = self.decoder(x, hidden, cell, encoder_outputs)

            outputs[word] = output

            # get the best predicted word
            best = output.argmax(1)

            x = target[word] if random.random() < teacher_force_ratio else best

        return outputs


def main():
    train_data, valid_data, test_data = Multi30k.splits(
        exts=(".de", ".en"), fields=(germanField, englishField)
    )

    englishField.build_vocab(train_data, max_size=10000, min_freq=2)
    germanField.build_vocab(train_data, max_size=10000, min_freq=2)

    NUM_EPOCHS = 50
    LR = 3e-3
    BATCH_SIZE = 512

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size_encoder = len(germanField.vocab)
    input_size_decoder = len(englishField.vocab)
    output_size = len(englishField.vocab)
    encoder_emb_size = 256
    decoder_emb_size = 256
    hidden_size = 512  # must be same for both LSTMs
    num_layers = 1
    dropout = 0.5
    bidirectional = False

    writer = SummaryWriter()
    step = 0

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=BATCH_SIZE,
        sort_within_batch=True,
        sort_key=lambda x: len(x.src),
        device=device,
    )

    encoder = Encoder(
        input_size_encoder, encoder_emb_size, hidden_size, num_layers, dropout
    ).to(device)
    attn = Attention(hidden_size, hidden_size)
    decoder = Decoder(
        input_size_decoder,
        decoder_emb_size,
        hidden_size,
        output_size,
        num_layers,
        dropout,
        attn,
    ).to(device)
    model = Seq2Seq(encoder, decoder).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    pad_idx = englishField.vocab.stoi["<pad>"]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    sentence = " ".join(valid_data[1].src)
    print(" ".join(valid_data[1].src))
    print(" ".join(valid_data[1].trg))

    for epoch in tqdm(range(NUM_EPOCHS)):
        print(f"[Epoch {epoch} / {NUM_EPOCHS}]")

        epoch_loss = 0

        for batch_idx, batch in enumerate(train_iterator):
            src_data = batch.src.to(device)
            trg_data = batch.trg.to(device)

            output = model(src_data, trg_data)
            # output shape: (tgr_len, BATCH_SIZE, output_dim)

            output = output[1:].reshape(-1, output.shape[2])
            target = trg_data[1:].reshape(-1)

            optimizer.zero_grad()
            loss = criterion(output, target)
            loss.backward()

            epoch_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

            optimizer.step()

        writer.add_scalar(
            "Training loss", epoch_loss / len(train_iterator), global_step=step
        )
        writer.add_scalar(
            "Evaluating loss",
            evaluate(model, valid_iterator, criterion),
            global_step=step,
        )

        model.eval()
        with torch.no_grad():
            translated_sentence = translate_sentence(
                model, sentence, germanField, englishField, device, max_length=50
            )
        writer.add_text(
            "Translated example sentence",
            " ".join(translated_sentence),
            global_step=epoch,
        )

        if (step + 1) % 10 == 0:
            bleu_score = bleu(test_data[:100], model, germanField, englishField, device)
            writer.add_scalar("BLEU score", bleu_score, global_step=step)

        model.train()
        step += 1


if __name__ == "__main__":
    main()
