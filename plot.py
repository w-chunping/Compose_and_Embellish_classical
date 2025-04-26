import re
import matplotlib.pyplot as plt

# Change this to your file path
filename = './ckpt/stage02_embellish_gp_m040_new/valloss.txt'

# Lists to store the extracted values
epochs = []
train_losses = []
val_losses = []

# Regex pattern to match the required parts
pattern = re.compile(r'ep(\d+)\s+\|\s+loss:\s+([\d.]+)\s+valloss:\s+([\d.]+)')

# Read and parse the file
with open(filename, 'r') as file:
    for line in file:
        match = pattern.search(line)
        if match:
            epoch = int(match.group(1))
            train_loss = float(match.group(2))
            val_loss = float(match.group(3))
            
            epochs.append(epoch)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

# Plotting the losses
plt.figure(figsize=(12, 6))
plt.plot(epochs, train_losses, label='Training Loss', marker='o', linewidth=1.5)
plt.plot(epochs, val_losses, label='Validation Loss', marker='x', linewidth=1.5)
plt.title('Training and Validation Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('m040_new_loss_plot.png')
print("Plot saved as loss_plot.png")
