import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
import random
import pandas as pd
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader

from prutils.prdataset import MBTIDataset, EssaysDataset
from prutils.prevaluation import PrEvaluation, print_performance
from prutils.prtracking import TrackingManager

from transformers import BertModel

# define a fucntion reset random seed
def reset_random_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

reset_random_seed()

# check cuda
device = "cuda" if torch.cuda.is_available() else "cpu"



class PersonalityClassifier(nn.Module):
    def __init__(self, language_encoder, number_of_classes):
        super(PersonalityClassifier, self).__init__()
        self.language_encoder = language_encoder

        # Define the 2-layer MLP
        self.mlp = nn.Sequential(
            nn.Linear(self.language_encoder.config.hidden_size, 128),  # First layer with 128 hidden units
            nn.ReLU(),  # Activation function
            nn.Linear(128, number_of_classes)  # Output layer
        )

    def forward(self, input_ids, attention_mask=None):
        # Get language model outputs
        language_encoding = self.language_encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Take the output of the [CLS] token
        personality_embeddings = language_encoding.last_hidden_state[:, 0, :]
        
        # Pass through the 2-layer MLP
        logits = self.mlp(personality_embeddings)
        
        return logits



def train_step(tracker, train_loader, personality_classifier, loss_fn, optimizer, scheduler):
    personality_classifier.train()
    for input_ids,attention_mask,labels in train_loader:
        optimizer.zero_grad()
        logits = personality_classifier(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(logits, labels)
        tracker.training_push(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(personality_classifier.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

def validation_step(tracker, validation_loader, personality_classifier, loss_fn):
    personality_classifier.eval()
    for input_ids,  attention_mask, labels in validation_loader:
        with torch.no_grad():
            logits = personality_classifier(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(logits, labels)
            tracker.validation_push(loss.item())

def evaluation_step(evaluation_loader, personality_classifier, loss_fn):
    personality_classifier.eval()
    evaluater = PrEvaluation()
    for input_ids, attention_mask, labels in evaluation_loader:
        with torch.no_grad():
            logits = personality_classifier(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.nn.functional.sigmoid(logits)
            predictions = (probs > 0.5).float()

            # rearrange accorind to dimension
            predictions = predictions.T
            labels = labels.T

            predictions = predictions.cpu().numpy().tolist()
            labels = labels.cpu().numpy().tolist()
            evaluater.push(predictions, labels)
    return evaluater.get_performance_metrics()


def run(dataset_name, model_name):
    print(f"Running {dataset_name} ")
    # Initialize the tokenizer and model
    language_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/'+model_name)
    language_encoder = AutoModel.from_pretrained('sentence-transformers/'+model_name)
    #load weight from 'contrastive_model_weights.pth'
    language_encoder.load_state_dict(torch.load('essays_test_model_weights.pth'))

    if dataset_name == "mbti":
        epochs = 10
        trainset = MBTIDataset(device=device, settype="train", tokenizer=language_tokenizer)
        validationset = MBTIDataset(device=device, settype="validation", tokenizer=language_tokenizer)
        evaluationset = MBTIDataset(device=device, settype="evaluation", tokenizer=language_tokenizer)
        number_of_classes=4
    else:
        epochs = 10
        trainset = EssaysDataset(device=device, settype="train", tokenizer=language_tokenizer)
        validationset = EssaysDataset(device=device, settype="validation", tokenizer=language_tokenizer)
        evaluationset = EssaysDataset(device=device, settype="evaluation", tokenizer=language_tokenizer)

        # mix trainset and 80% evaluationset randomly
      
        number_of_classes=5

    train_loader = DataLoader(trainset, batch_size=32, shuffle=True)
    validation_loader = DataLoader(validationset, batch_size=32, shuffle=False)
    evaluation_loader = DataLoader(evaluationset, batch_size=32, shuffle=False)
    print(f"Data loaded: {len(trainset)} training samples, {len(validationset)} validation samples, {len(evaluationset)} evaluation samples")

    personality_classifier = PersonalityClassifier(language_encoder = language_encoder, number_of_classes=number_of_classes)
    personality_classifier.to(device)

    loss_fn = torch.nn.BCEWithLogitsLoss()
    # if torch.cuda.device_count() > 1:
    #     print("Using", torch.cuda.device_count(), "GPUs!")
    #     personality_classifier = nn.DataParallel(personality_classifier)

    from transformers import AdamW

    # Freeze encoder parameters
    for name, param in personality_classifier.named_parameters():
        if "encoder" in name:  # Adjust "encoder" to match the exact name of your encoder module
            param.requires_grad = False

    # Define optimizer for only trainable parameters
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, personality_classifier.parameters()),
        lr=5e-5
    )

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps = 0.1,
                                                num_training_steps=len(train_loader)*epochs)


    tracker = TrackingManager(f"{dataset_name}/{model_name}/")

    for epoch in range(epochs):
        train_step(tracker, train_loader, personality_classifier, loss_fn, optimizer, scheduler)
        validation_step(tracker, validation_loader, personality_classifier, loss_fn)
        performance = evaluation_step(evaluation_loader, personality_classifier, loss_fn)

        info = tracker.go_to_new_epoch(personality_classifier)
        tl = info["training_loss"]
        vl = info["validation_loss"]
        mark = info["mark"]

        print(f"Epoch {epoch+1}/{epochs} {mark} - TL:{tl:.4f}, VL:{vl:4f}")
        print_performance(performance)

    #save
    save_path = f"{dataset_name}_{model_name}_classify.pth"
    torch.save(personality_classifier.language_encoder.state_dict(), save_path)

        


if __name__ == "__main__":
    name = '_____'
    #run("mbti", name)
    run("essays", name)
    

