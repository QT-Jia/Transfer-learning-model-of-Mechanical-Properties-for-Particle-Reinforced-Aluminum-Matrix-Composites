import torch
from sklearn.preprocessing import StandardScaler
import numpy as np

s_elec_dict = {'Si': 2, 'Fe': 2, 'Cu': 1, 'B': 2, 'Zn': 2, 'Mn': 2, 'Mg': 2, 'Ti': 2,'V': 2, 'Ni':2,
               'Ce':2, 'Cr':1, 'Sc':2, 'Sr':2, 'Zr':2, 'Li':1, 'Al':2}

p_elec_dict = {'Si':2, 'Fe':6, 'Cu':0, 'B':1, 'Zn':0, 'Mn':0, 'Mg':6, 'Ti':0,'V':0, 'Ni':0,
               'Ce':6, 'Cr':0, 'Sc':0, 'Sr':6, 'Zr':0, 'Li':0,'Al':1}

d_elec_dict = {'Si':0, 'Fe':6, 'Cu':10, 'B':0, 'Zn':10, 'Mn':5, 'Mg':0, 'Ti':2,'V':3, 'Ni':8,
               'Ce':1, 'Cr':5, 'Sc':1, 'Sr':0, 'Zr':2, 'Li':0, 'Al':0}

radius_dict = {'Si':1.17, 'Fe':1.56, 'Cu':1.45, 'B':0.87, 'Zn':1.53, 'Mn':1.79, 'Mg':1.72, 'Ti':2.0,'V':1.53, 'Ni':1.51,
               'Ce':1.83, 'Cr':1.56, 'Sc':1.84, 'Sr':2.15, 'Zr':2.16, 'Li':1.82, 'Al':1.43}

atom_mass_dict = {'Si':28.09, 'Fe':55.85, 'Cu':63.55, 'B':10.81, 'Zn':65.39, 'Mn':54.94, 'Mg':24.31, 'Ti':47.90,'V':50.90, 'Ni':58.69,
             'Ce':140.12, 'Cr':52.00, 'Sc':44.96, 'Sr':87.62, 'Zr':91.22, 'Li':6.94, 'Al':26.98}

elec_gativity_dict = {'Si':1.90, 'Fe':1.83, 'Cu':1.90, 'B':2.04, 'Zn':1.65, 'Mn':1.55, 'Mg':1.31, 'Ti':1.54,'V':1.63, 'Ni':1.91,
                          'Ce':1.12, 'Cr':1.66, 'Sc':1.36, 'Sr':0.95, 'Zr':1.33, 'Li':0.98, 'Al':1.61}




class Feature_Fusion:
    def __init__(self,at_value, length, process_value):
        self.at_value = at_value
        self.process_value = process_value
        self.length = length

    def scaler(self, feature):
        feature_array = np.array(list(feature))
        return feature_array / np.max(np.abs(feature_array))
    
    def c_embedding(self):
        c = torch.zeros(self.length, 16, 6)
        for i in range(self.length):
            c[i, :, 0: 6] = self.at_value[i].t().reshape(-1, 1).repeat(1, 6)
        return c

    def p_embedding(self):
        p = torch.zeros(self.length, 17, 6)
        for i in range(self.length):
            p[i, :, 0] = torch.from_numpy(self.scaler(s_elec_dict.values())).type(torch.float).t()
            p[i, :, 1] = torch.from_numpy(self.scaler(p_elec_dict.values())).type(torch.float).t()
            p[i, :, 2] = torch.from_numpy(self.scaler(d_elec_dict.values())).type(torch.float).t()
            p[i, :, 3] = torch.from_numpy(self.scaler(radius_dict.values())).type(torch.float).t()
            p[i, :, 4] = torch.from_numpy(self.scaler(atom_mass_dict.values())).type(torch.float).t()
            p[i, :, 5] = torch.from_numpy(self.scaler(elec_gativity_dict.values())).type(torch.float).t()
        return p
    
    def fit(self):
        attention_data = torch.zeros(self.length, 2, 17, 6)
        attention_data[:, 0, :, :] = self.c_embedding()
        attention_data[:, 1, :, :] = self.p_embedding()

        scaler = StandardScaler()
        scaler_values = scaler.fit_transform(self.process_value)
        scaler_values = torch.from_numpy(scaler_values).type(torch.float)
        return [attention_data, scaler_values]
    


class CustomNormalizer:
    def __init__(self):
        self.means = []
        self.stds = []
    
    def fit_transform(self, data):
        if not isinstance(data, torch.Tensor):
            data = torch.from_numpy(data).float()
        if len(data.shape) == 1:
            data = data.view(-1, 1)    
            
        normalized_data = data.clone()
        self.means = []
        self.stds = []
        
        for i in range(data.shape[1]):
            mask = data[:, i] != 0
            if mask.sum() > 0:
                mean = data[mask, i].mean()
                std = data[mask, i].std()
                normalized_data[mask, i] = (data[mask, i] - mean) / std

                self.means.append(mean.item())
                self.stds.append(std.item())
            else:
                self.means.append(0.0)
                self.stds.append(1.0)

        self.means = torch.tensor(self.means)
        self.stds = torch.tensor(self.stds)
        
        return normalized_data
    
    def inverse_transform(self, data):
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()

        if not isinstance(self.means, torch.Tensor):
            self.means = torch.tensor(self.means)
        if not isinstance(self.stds, torch.Tensor):
            self.stds = torch.tensor(self.stds)
        device = data.device
        self.means = self.means.to(device)
        self.stds = self.stds.to(device)

        denormalized_data = data.clone()
        
        for i in range(data.shape[1]):
            mask = data[:, i] != 0
            if mask.sum() > 0:
                denormalized_data[mask, i] = data[mask, i] * self.stds[i] + self.means[i]

        if isinstance(data, np.ndarray):
            return denormalized_data.cpu().numpy()
        
        return denormalized_data