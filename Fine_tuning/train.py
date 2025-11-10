import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from DStream_model import MaskedMSELoss

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_data(device):
    attention_data = torch.load('./attention_data_AlMC.pt', map_location=device)
    scaler_values = torch.load('./scaler_values_AlMC.pt', map_location=device)
    particle_data = torch.load('./particle_data_ALMC.pt', map_location=device)
    attention_data = attention_data.float()
    scaler_values = scaler_values.float()
    particle_data = particle_data.float()

    df = pd.read_excel('./composite_data.xlsx', sheet_name='Sheet1')
    output_data = df.iloc[:, -2:].values
    output_data = torch.FloatTensor(output_data)


    return [attention_data, scaler_values, particle_data], output_data.to(device)

def split_data(data, output_data, device, test_size=0.2):
    attention_data, scaler_values, particle_data = data

    indices = torch.arange(len(attention_data), device=device)
    n_test = int(len(indices) * test_size)
    shuffled_indices = torch.randperm(len(indices), device=device)
    test_indices = shuffled_indices[:n_test]
    train_indices = shuffled_indices[n_test:]

    train_attention = attention_data[train_indices]
    test_attention = attention_data[test_indices]

    train_scaler = scaler_values[train_indices]
    test_scaler = scaler_values[test_indices]
    
    train_particle = particle_data[train_indices]
    test_particle = particle_data[test_indices]

    train_input = [train_attention, train_scaler, train_particle]
    test_input = [test_attention, test_scaler, test_particle]

    train_output = output_data[train_indices]
    test_output = output_data[test_indices]

    return (train_input, train_output), (test_input, test_output), train_indices.cpu(), test_indices.cpu()


def to_device(data, device):
    if isinstance(data, (tuple, list)):
        return tuple(to_device(x, device) for x in data)
    return data.to(device)


def evaluate_model(model, input_data, true_output,output_normalizer,device):
    model.eval()
    with torch.no_grad():
        input_data = [x.to(device) for x in input_data if isinstance(x, torch.Tensor)]
        true_output = true_output.to(device)
        pred_output = model(input_data)

        means = torch.load('./normalization_output.pt', weights_only=True)['means'].to(device)
        stds = torch.load('./normalization_output.pt', weights_only=True)['stds'].to(device)
        pred_denorm = pred_output * stds + means
        true_output_denorm = output_normalizer.inverse_transform(true_output)

        pred_np = pred_denorm.cpu().numpy()
        true_values_np = true_output_denorm.cpu().numpy()

        metrics = []
        for i in range(2):
            mask = true_values_np[:, i] != 0
            if mask.sum() > 0:
                r2 = r2_score(true_values_np[mask, i], pred_np[mask, i])
                mse = mean_squared_error(true_values_np[mask, i], pred_np[mask, i])
                mae = mean_absolute_error(true_values_np[mask, i], pred_np[mask, i])
                metrics.append((r2, mse, mae))
            else:
                metrics.append((np.nan, np.nan, np.nan))


        val_loss = mean_squared_error(true_values_np, pred_np)

    return val_loss, metrics, pred_denorm


def train_model(model, train_data, output_normalizer, device, epochs, batch_size, lr, weight_ratio):
    model = model.to(device)  
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = float('inf')
    best_model_state = None

    train_input, train_output = train_data
    train_input = [x.to(device) for x in train_input]
    train_output = train_output.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        indices = torch.randperm(len(train_input[0]), device=device)
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]


            batch_input = [x[batch_indices] for x in train_input]
            batch_output = train_output[batch_indices]

            optimizer.zero_grad()
            predictions = model(batch_input)


            loss_fn = MaskedMSELoss(weight_ratio=weight_ratio)
            main_loss = loss_fn(predictions, batch_output)
            main_loss.backward()
            optimizer.step()
            total_loss += main_loss.item()

        avg_loss = total_loss / len(train_input[0])

        if (epoch + 1) % 200 == 0:
            model.eval()
            with torch.no_grad():
                val_loss, metrics, _ = evaluate_model(model, train_input, train_output, output_normalizer, device)

            model.train()

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_state = {k: v.detach().clone() for k, v in model.state_dict().items()}


    return best_model_state