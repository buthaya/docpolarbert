import torch

from torch.utils.data import DataLoader
from seqeval.metrics import (f1_score, precision_score, recall_score)

from src.modeling.docpolarbert.modeling_docpolarbert import DocPolarBERTForTokenClassification

# ------------------------------------------------------------------------- #
def train_step_token_classification(
        model: DocPolarBERTForTokenClassification,
        train_dataloader: DataLoader,
        idx2label: dict,
        optimizer,
        ):
    model.train()

    batch_idx = 0
    total_loss = 0

    preds = []
    out_label_ids= []

    for local_step, batch in enumerate(train_dataloader):
        # ------------------------------------ Data preparation ------------------------------------ #
        input_ids = batch["input_ids"].to(model.device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(model.device, non_blocking=True)
        token_type_ids = batch["token_type_ids"].to(model.device, non_blocking=True)
        labels = batch["labels"].to(model.device, non_blocking=True)
        bbox = batch["bbox"].to(model.device, non_blocking=True)

        # ------------------------------------ Forward pass ------------------------------------ #
        outputs = model(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            return_dict=None,
        )

        loss = outputs.loss
        loss = loss   # Normalize loss
        logits = outputs.logits

        # Ignore -100 index
        mask = (labels != -100)
        valid_preds = torch.argmax(logits, dim=2)
        valid_preds = valid_preds.masked_select(mask)
        valid_labels = labels.masked_select(mask)

        preds.append([idx2label[p.item()] for p in valid_preds])
        out_label_ids.append([idx2label[l.item()] for l in valid_labels])

        # ------------------------- Backward pass to get the gradients ------------------------- #
        loss.backward()
        # ---------------------------------- Optimizer update ---------------------------------- #
        optimizer.step()  # Update model parameters
        optimizer.zero_grad()  # Reset gradients

        total_loss += loss.item()
        batch_idx += 1

    avg_loss = total_loss / batch_idx

    return avg_loss

def eval_token_classification(
        model: DocPolarBERTForTokenClassification,
        eval_dataloader: DataLoader,
        idx2label: dict,
        print_results=True):
    model.eval()

    total_loss = 0.0
    nb_eval_steps = 0
    preds = []
    out_label_ids = []

    iterator = eval_dataloader
    for batch in iterator:
        with torch.no_grad():
            input_ids = batch["input_ids"].to(model.device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(model.device, non_blocking=True)
            token_type_ids = batch["token_type_ids"].to(model.device, non_blocking=True)
            labels = batch["labels"].to(model.device, non_blocking=True)
            bbox = batch["bbox"].to(model.device, non_blocking=True)

            # ------------------------------------ Forward pass ------------------------------------ #
            outputs = model(
                input_ids=input_ids,
                bbox=bbox,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                output_attentions=None,
                return_dict=None,
            )

            # get the loss and logits
            loss = outputs.loss
            logits = outputs.logits

            # Ignore -100 index
            mask = (labels != -100)
            valid_preds = torch.argmax(logits, dim=2)
            valid_preds = valid_preds.masked_select(mask)
            valid_labels = labels.masked_select(mask)

            total_loss += loss.sum()
            nb_eval_steps += max(1, torch.cuda.device_count())

            preds.append([idx2label[p.item()] for p in valid_preds])
            out_label_ids.append([idx2label[l.item()] for l in valid_labels])

            precision = precision_score(out_label_ids, preds)
            recall = recall_score(out_label_ids, preds)
            f1 = f1_score(out_label_ids, preds)

    # compute average evaluation loss
    avg_loss = (total_loss / nb_eval_steps).item()

    return avg_loss, precision, recall, f1
