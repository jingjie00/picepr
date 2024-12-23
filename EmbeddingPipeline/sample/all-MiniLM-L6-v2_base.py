import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, AdamW
from prutils.prdataset import ContrastiveEssaysDataset, ContrastiveMBTIDataset


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for anchor_ids, anchor_mask, paired_ids, paired_mask, labels in dataloader:
        # Move data to device
        anchor_ids, anchor_mask = anchor_ids.to(device), anchor_mask.to(device)
        paired_ids, paired_mask = paired_ids.to(device), paired_mask.to(device)
        labels = labels.to(device)

        # Forward pass
        anchor_output = model(input_ids=anchor_ids, attention_mask=anchor_mask).pooler_output
        paired_output = model(input_ids=paired_ids, attention_mask=paired_mask).pooler_output

        # Compute cosine similarity
        similarity = torch.nn.functional.cosine_similarity(anchor_output, paired_output)

        # Contrastive loss
        loss = torch.nn.functional.binary_cross_entropy_with_logits(similarity, labels)
        total_loss += loss.item()

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return total_loss / len(dataloader)


def main(dataset,model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/"+model_name)
    model = AutoModel.from_pretrained("sentence-transformers/"+model_name).to(device)

    if dataset == "essays":
        # Prepare dataset and dataloader
        trainset = ContrastiveEssaysDataset(device=device, settype="train", tokenizer=tokenizer)
        trainloader = DataLoader(trainset, batch_size=16, shuffle=True)
    else:
        trainset = ContrastiveMBTIDataset(device=device, settype="train", tokenizer=tokenizer)
        trainloader = DataLoader(trainset, batch_size=16, shuffle=True)

    # Define optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # Training loop
    epochs = 20
    for epoch in range(epochs):
        avg_loss = train_epoch(model, trainloader, optimizer, device)
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")
    
        save_path = f"{dataset}_{model_name}_base.pth"

        # Save the model's state_dict (weights and biases)
        torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    main("mbti","all-MiniLM-L6-v2")
    #main("essays", "all-MiniLM-L6-v2")
