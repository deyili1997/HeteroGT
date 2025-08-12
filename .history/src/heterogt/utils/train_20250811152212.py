from copy import deepcopy

def train_with_early_stopping(model, train_dataloader, val_dataloader, test_dataloader,
                              optimizer, loss_fn, device, config, task_type, epochs,
                              val_long_seq_idx=None, test_long_seq_idx=None, eval_metric="prauc", return_model=False):

    best_score = 0.
    best_val_metric = None
    best_test_metric = None
    best_model_state = deepcopy(model.state_dict())
    epochs_no_improve = 0

    if model_name == "GT":
        reg_loss = GraphAttnRegularizer(config).to(device)

    scaler = GradScaler()

    for epoch in range(1, epochs + 1):
        model.train()
        avg_loss = avg_task_loss = avg_bias_loss = avg_ent_loss = 0.
        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch:03d}")

        for step, batch in progress_bar:
            optimizer.zero_grad()
            try:
                batch = [x.to(device) if isinstance(x, torch.Tensor) else x for x in batch]
                labels = batch[-1].float()

                with autocast():
                    if model_name == "GT":
                        prediction, _, attn_logits_list, occur_param = model(*batch[:-1])
                        task_loss = loss_fn(prediction.view(-1), labels.view(-1))
                        bias_loss, ent_loss = reg_loss(attn_logits_list, occur_param)
                        loss = task_loss + bias_loss + ent_loss
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                        avg_task_loss += task_loss.item()
                        avg_bias_loss += bias_loss.item()
                        avg_ent_loss += ent_loss.item()

                    elif model_name == "HG":
                        prediction, edge_embed = model(*batch[:-1])
                        loss = loss_fn(prediction.view(-1), labels.view(-1))
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        avg_task_loss += loss.item()
                    
                    elif model_name == "Concat_fuse":
                        pred_GT, pred_HG, pred_fuse = model(*batch[:-1])
                        loss_GT = loss_fn(pred_GT.view(-1), labels.view(-1))
                        loss_HG = loss_fn(pred_HG.view(-1), labels.view(-1))
                        loss_fuse = loss_fn(pred_fuse.view(-1), labels.view(-1))
                        loss = loss_GT + loss_HG + loss_fuse
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        avg_task_loss += loss.item()
                    
                    else:
                        raise ValueError(f"Unsupported model_name: {model_name}")

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                avg_loss += loss.item()
                num_steps = max(1, step + 1)
                current_loss = avg_loss / num_steps

                postfix_dict = {
                    "task_loss": f"{avg_task_loss / num_steps:.4f}",
                    "loss": f"{current_loss:.4f}"
                }
                if model_name == "GT":
                    postfix_dict.update({
                        "bias_loss": f"{avg_bias_loss / num_steps:.4f}",
                        "ent_loss": f"{avg_ent_loss / num_steps:.4f}"
                    })
                progress_bar.set_postfix(postfix_dict)

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"[OOM Warning] Skipping batch {step} due to CUDA OOM.")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e

        avg_loss /= max(1, step + 1)
        torch.cuda.empty_cache()

        print(f"Epoch: {epoch:03d}, Average Loss: {avg_loss:.4f}")

        # --- 复用抽象函数 ---
        (best_score, best_val_metric, best_test_metric, best_model_state,
         epochs_no_improve, early_stop_triggered) = evaluate_and_early_stop(
            model, val_dataloader, test_dataloader, device, task_type,
            val_long_seq_idx, test_long_seq_idx, eval_metric,
            best_score, best_val_metric, best_test_metric, best_model_state,
            epochs_no_improve, config["early_stop_patience"]
        )

        if early_stop_triggered:
            break

        # --- Smoothness Tracking for HG ---
        if model_name == "HG":
            model.eval()
            all_edge_embeds = []
            with torch.no_grad():
                for val_batch in val_dataloader:
                    val_batch = [x.to(device) if isinstance(x, torch.Tensor) else x for x in val_batch]
                    _, edge_embed = model(*val_batch[:-1])
                    all_edge_embeds.append(edge_embed.cpu())

            edge_embeds = torch.cat(all_edge_embeds, dim=0)
            norm_embed = F.normalize(edge_embeds, dim=1)
            sim_matrix = norm_embed @ norm_embed.T
            sim_vals = sim_matrix[~torch.eye(sim_matrix.size(0), dtype=bool)]
            mean_cos_sim = sim_vals.mean().item()
            embed_var = edge_embeds.var(dim=0).mean().item()

            print(f"[Smoothness Check] Mean Cosine Similarity: {mean_cos_sim:.4f} | "
                  f"Mean Embedding Variance: {embed_var:.4f}")

    print("\nBest validation performance:")
    print(best_val_metric)
    print("Corresponding test performance:")
    print(best_test_metric)

    # Optional: restore best model before returning
    model.load_state_dict(best_model_state)
    return (best_test_metric, model) if return_model else best_test_metric


def evaluate_and_early_stop(model, val_dataloader, test_dataloader, device, task_type,
                                  val_long_seq_idx, test_long_seq_idx, eval_metric,
                                  best_score, best_val_metric, best_test_metric, best_model_state,
                                  epochs_no_improve, early_stop_patience):
    """
    执行模型在验证集和测试集的评估，并进行早停检查。
    返回：
        - best_score
        - best_val_metric
        - best_test_metric
        - best_model_state
        - epochs_no_improve
        - early_stop_triggered (bool)
    """
    # --- Evaluation ---
    if val_long_seq_idx is not None:
        val_metric, val_long_seq_metric = evaluate(model, val_dataloader, device, task_type, val_long_seq_idx)
    else:
        val_metric = evaluate(model, val_dataloader, device, task_type)
        val_long_seq_metric = None

    if test_long_seq_idx is not None:
        test_metric, test_long_seq_metric = evaluate(model, test_dataloader, device, task_type, test_long_seq_idx)
    else:
        test_metric = evaluate(model, test_dataloader, device, task_type)
        test_long_seq_metric = None

    print(f"Validation: {val_metric}")
    print(f"Validation-long: {val_long_seq_metric if val_long_seq_metric is not None else 'Not provided'}")
    print(f"Test:      {test_metric}")
    print(f"Test-long: {test_long_seq_metric if test_long_seq_metric is not None else 'Not provided'}")

    # --- Early Stopping ---
    current_score = val_metric[eval_metric]
    early_stop_triggered = False

    if current_score > best_score:
        best_score = current_score
        best_val_metric = val_metric
        best_test_metric = test_metric
        best_model_state = deepcopy(model.state_dict())
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= early_stop_patience:
            print(f"\nEarly stopping triggered (no improvement for {early_stop_patience} epochs).")
            early_stop_triggered = True

    return best_score, best_val_metric, best_test_metric, best_model_state, epochs_no_improve, early_stop_triggered


def run_multilabel_metrics(predictions, labels):
    # Multi-label classification: predictions [B, C], labels [B, C]
    f1s, aucs, praucs, precisions, recalls = [], [], [], [], []

    for i in range(predictions.size(0)):
        pred_i = predictions[i].clone()
        label_i = labels[i].clone()

        pred_i = (pred_i > 0).float().numpy()
        label_i = label_i.float().numpy()

        tp = (pred_i * label_i).sum()
        precision = tp / (pred_i.sum() + 1e-8)
        recall = tp / (label_i.sum() + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        try:
            auc_score = roc_auc_score(label_i, pred_i)
        except ValueError:
            auc_score = np.nan  # skip if only one class present

        prec_curve, rec_curve, _ = precision_recall_curve(label_i, pred_i)
        pr_auc_score = auc(rec_curve, prec_curve)

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        aucs.append(auc_score)
        praucs.append(pr_auc_score)

    return {
        "precision": np.nanmean(precisions),
        "recall": np.nanmean(recalls),
        "f1": np.nanmean(f1s),
        "auc": np.nanmean(aucs),
        "prauc": np.nanmean(praucs),
    }


@torch.no_grad()
def evaluate(model, dataloader, device, task_type, long_seq_idx=None):
    model.eval()
    all_preds, all_labels = [], []
    
    for _, batch in enumerate(tqdm(dataloader, desc="Running inference")):
        batch = [x.to(device) if isinstance(x, torch.Tensor) else x for x in batch]
        labels = batch[-1]
        with autocast():
            output = model(*batch[:-1])

        # Compatible with outputs as tensor or tuple/list
        preds = output[0] if isinstance(output, (tuple, list)) else output

        all_preds.append(preds)
        all_labels.append(labels)

    predictions = torch.cat(all_preds, dim=0).cpu()
    labels = torch.cat(all_labels, dim=0).cpu()

    if task_type == "binary":
        binary_task_results = run_binary_metrics(predictions, labels)
        if long_seq_idx is not None:
            long_seq_labels = labels[long_seq_idx]
            long_seq_preds = predictions[long_seq_idx]
            long_seq_results = run_binary_metrics(long_seq_preds, long_seq_labels)
            return binary_task_results, long_seq_results
        else:
            return binary_task_results

    else:
        multilabel_task_results = run_multilabel_metrics(predictions, labels)    
        if long_seq_idx is not None:
            long_seq_labels = labels[long_seq_idx]
            long_seq_preds = predictions[long_seq_idx]
            long_seq_results = run_multilabel_metrics(long_seq_preds, long_seq_labels)
            return multilabel_task_results, long_seq_results
        else:
            return multilabel_task_results