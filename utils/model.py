from torch import nn
import torch

# Alfabeto posible (mayúsculas, dígitos, guión)
CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
NUM_CLASSES = len(CHARS) + 1  # +1 para el token en blanco de CTC

char_to_idx = {char: idx for idx, char in enumerate(CHARS)}
idx_to_char = {idx: char for char, idx in char_to_idx.items()}

class CRNN(nn.Module):
    def __init__(self, img_h, num_classes):
        super(CRNN, self).__init__()

        # Backbone CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # (C,H,W)
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # H/2

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # H/4

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1)),  # H/8

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1)),  # H/16

            nn.Conv2d(512, 512, kernel_size=2),
            nn.ReLU(True)
        )

        # RNN para procesar secuencia
        self.rnn = nn.Sequential(
            nn.LSTM(512, 256, bidirectional=True, batch_first=True),
            nn.LSTM(512, 256, bidirectional=True, batch_first=True)
        )

        # Capa final
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # CNN -> características
        conv = self.cnn(x)  # (B, C, H', W')

        # Ajustar para RNN (B, W', C*H')
        b, c, h, w = conv.size()
        conv = conv.permute(0, 3, 1, 2)  # (B, W', C, H')
        conv = conv.view(b, w, c * h)

        # Pasar por RNN
        recurrent, _ = self.rnn(conv)

        # Clasificación por timestep
        output = self.fc(recurrent)  # (B, W', num_classes)
        output = output.permute(1, 0, 2)  # CTC Loss espera (T, B, num_classes)
        return output

if __name__ == "__main__":
    # Alfabeto posible (mayúsculas, dígitos, guión)
    CHARS = "+-0123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    NUM_CLASSES = len(CHARS) + 1  # +1 para el token en blanco de CTC

    char_to_idx = {char: idx for idx, char in enumerate(CHARS)}
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    """
    # Parámetros
    IMG_SIZE = 128  # normalizamos todas las imágenes a altura fija
    model = CRNN(IMG_SIZE, NUM_CLASSES)

    # Ejemplo batch (B=2, 3 canales, H=32, W=100)
    dummy_input = torch.randn(2, 3, IMG_SIZE, IMG_SIZE)
    out = model(dummy_input)
    print(out.shape)  # (T, B, num_classes)

    ctc_loss = nn.CTCLoss(blank=NUM_CLASSES - 1, zero_infinity=True)

    # Ejemplo de datos simulados
    preds = out.log_softmax(2)  # CTC requiere log-softmax
    input_lengths = torch.full(size=(2,), fill_value=preds.size(0), dtype=torch.long)  # longitudes predicciones
    target_texts = ["AB-12", "X9Z"]  # ejemplo

    # Codificar targets
    targets = []
    target_lengths = []
    for txt in target_texts:
        encoded = [char_to_idx[c] for c in txt]
        targets.extend(encoded)
        target_lengths.append(len(encoded))
    targets = torch.tensor(targets, dtype=torch.long)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)

    # Calcular pérdida
    loss = ctc_loss(preds, targets, input_lengths, target_lengths)
    loss.backward()


    def ctc_decode(preds):
        # preds: (T, B, num_classes)
        preds_idx = preds.argmax(2).permute(1, 0)  # (B, T)
        results = []
        for seq in preds_idx:
            text = ""
            prev = -1
            for idx in seq:
                idx = idx.item()
                if idx != prev and idx != NUM_CLASSES - 1:  # ignorar repetidos y blank
                    text += idx_to_char[idx]
                prev = idx
            results.append(text)
        return results

    ctc_decode(preds)
"""