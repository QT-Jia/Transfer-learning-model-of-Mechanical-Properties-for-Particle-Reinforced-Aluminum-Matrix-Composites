import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score, mean_squared_error
import torch.serialization
from model import MaskedMSELoss
from Embedding_alloy import CustomNormalizer


# 设置随机种子
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True



def load_data(device):
    torch.serialization.add_safe_globals(
        {'torch': torch, 'numpy': np}
    )

    attention_data = torch.load(
        'D:/ML_AMC/Test/Data_clean/Result_410/Attention_data.pt',
         weights_only=True, map_location=device
    )

    scaler_values = torch.load(
        'D:/ML_AMC/Test/Data_clean/Result_410/Scaler_values.pt',
        weights_only=True, map_location=device
    )
    attention_data = attention_data.float()
    scaler_values = scaler_values.float()

    df = pd.read_excel('./data.xlsx', sheet_name='Sheet1')
    df = df.fillna(0)
    output_data = df.iloc[:, -4:-1].astype(float).values
    output_data = torch.FloatTensor(output_data)

    normalizer = CustomNormalizer()
    normalized_output = normalizer.fit_transform(output_data)
    torch.save({
        'means': normalizer.means,
        'stds': normalizer.stds
    }, './Norm_Output.pt')

    return [attention_data, scaler_values], normalized_output.to(device), normalizer



def split_data(data, output_data, test_size=0.2):
    attention_data, scaler_values = data
    device = attention_data.device


    indices = torch.arange(len(attention_data), device=device)
    n_test = int(len(indices) * test_size)
    shuffled_indices = torch.randperm(len(indices), device=device)
    test_indices = shuffled_indices[:n_test]
    train_indices = shuffled_indices[n_test:]
    train_attention = attention_data[train_indices]
    test_attention = attention_data[test_indices]

    train_scaler = scaler_values[train_indices]
    test_scaler = scaler_values[test_indices]

    train_output = output_data[train_indices]
    test_output = output_data[test_indices]

    train_data = [train_attention, train_scaler]
    test_data = [test_attention, test_scaler]

    return (train_data, train_output), (test_data, test_output), train_indices.cpu(), test_indices.cpu()


def to_device(data, device):
    if isinstance(data, (tuple, list)):
        return tuple(to_device(x, device) for x in data)
    return data.to(device)


def evaluate_model(model, test_data, device, normalizer):
    model.eval()
    with torch.no_grad():
        test_input, test_output = test_data
        test_input = [x.to(device) for x in test_input]
        test_output = test_output.to(device)

        pred = model(test_input)
        pred_cpu = pred.cpu()
        true_values_cpu = test_output.cpu()

        pred_denorm = normalizer.inverse_transform(pred_cpu)
        true_values_denorm = normalizer.inverse_transform(true_values_cpu)
        pred_np = pred_denorm.numpy()
        true_values_np = true_values_denorm.numpy()

        metrics = []
        num_outputs = true_values_np.shape[1]

        for i in range(num_outputs):
            mask = true_values_np[:, i] != 0
            if mask.sum() > 0:
                r2 = r2_score(true_values_np[mask, i], pred_np[mask, i])
                mse = mean_squared_error(true_values_np[mask, i], pred_np[mask, i])
                metrics.append((r2, mse))

        val_loss = mean_squared_error(true_values_np, pred_np)

    return val_loss, metrics,pred_denorm.numpy()


def train_model(model, train_data, test_data, device, normalizer, epochs=500, batch_size=128, lr=0.005, logger=None):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = float('inf')
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        train_input, train_output = train_data
        train_input = [x.to(device) for x in train_input]
        train_output = train_output.to(device)

        indices = torch.randperm(len(train_input[0]))
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]

            batch_input = [x[batch_indices] for x in train_input]
            batch_output = train_output[batch_indices]

            optimizer.zero_grad()
            predictions = model(batch_input)

            loss_fn = MaskedMSELoss()
            loss = loss_fn(predictions, batch_output)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_input[0])
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    return best_model_state
