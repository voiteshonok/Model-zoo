from torchtext.data.metrics import bleu_score
import torch
import de_core_news_md


def bleu(data, model, german, english, device):
    targets = []
    outputs = []

    for example in data:
        src = vars(example)["src"]
        trg = vars(example)["trg"]

        prediction = translate_sentence(model, src, german, english, device)
        prediction = prediction[:-1]  # remove <eos> token

        targets.append([trg])
        outputs.append(prediction)

    return bleu_score(outputs, targets)


def translate_sentence(model, src, german, english, device, max_length=50):
    # Load german tokenizer
    spacy_ger = de_core_news_md.load()

    if type(src) == list:
        tokens = [token.text.lower() for token in spacy_ger(" ".join(src))]
    else:
        tokens = [token.text.lower() for token in spacy_ger(src)]

    # Add <SOS> and <EOS> in beginning and end respectively
    tokens.insert(0, german.init_token)
    tokens.append(german.eos_token)

    text_to_indices = [german.vocab.stoi[token] for token in tokens]

    # Convert to Tensor
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

    # Build encoder hidden, cell state
    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(sentence_tensor)

    outputs = [english.vocab.stoi["<sos>"]]

    for _ in range(max_length):
        previous_word = torch.LongTensor([outputs[-1]]).to(device)

        with torch.no_grad():
            output, hidden, cell = model.decoder(
                previous_word, hidden, cell, encoder_outputs
            )
            best = output.argmax(1).item()

            outputs.append(best)

            if output.argmax(1).item() == english.vocab.stoi["<eos>"]:
                break

    translated_sentence = [english.vocab.itos[idx] for idx in outputs]

    # remove start token
    return translated_sentence[1:]
