import torch
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

s_elec_dict = {'Si':2, 'Fe':2, 'Cu':1, 'B':2, 'Zn':2, 'Mn':2, 'Mg':2, 'Ti':2,'V':2, 'Ni':2,
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

hardness_dict = {'SiC':2840, 'TiC':2600, 'TiCN':2800, 'Al3Ti':400, 'ZrB2':2300, 'TiB2':3000}

meltpoint_dict = {'SiC':2800, 'TiC':3140, 'TiCN':3050, 'Al3Ti':1610, 'ZrB2':3245, 'TiB2':2980}

density_dict = {'SiC':3.20, 'TiC':4.93, 'TiCN':5.08, 'Al3Ti':3.36, 'ZrB2':5.80 ,'TiB2':4.59}

E_modulus_dict = {'SiC':537, 'TiC':425, 'TiCN':425, 'Al3Ti':171.7, 'ZrB2':614, 'TiB2':568}

thermal_misfit_dict = {'SiC':1.96, 'TiC':1.62, 'TiCN':1.76, 'Al3Ti':1.10, 'ZrB2':1.69, 'TiB2':1.56}


class Feature_Fusion:
    def __init__(self,at_value,particle_value, length, process_value):
        self.at_value = at_value
        self.particle_value = particle_value
        self.process_value = process_value
        self.length = length

    def scaler(self, feature):
        feature_array = np.array(list(feature))
        return feature_array / np.max(np.abs(feature_array))
    
    def c_embedding(self):
        c = torch.zeros(self.length, 17, 6)
        for i in range(self.length):
            c[i, :, 0: 6] = self.at_value[i].t().reshape(-1, 1).repeat(1, 6)
        return c

    def particle_embedding(self):
        pc = torch.zeros(self.length, 6, 5)
        for i in range(self.length):
            pc[i, :, 0: 5] = self.particle_value[i].t().reshape(-1, 1).repeat(1, 5)
        return pc

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
    
    def p_particle_embedding(self):
        p_pc = torch.zeros(self.length, 6, 5)
        for i in range(self.length):
            p_pc[i, :, 0] = torch.from_numpy(self.scaler(hardness_dict.values())).type(torch.float).t()
            p_pc[i, :, 1] = torch.from_numpy(self.scaler(meltpoint_dict.values())).type(torch.float).t()
            p_pc[i, :, 2] = torch.from_numpy(self.scaler(density_dict.values())).type(torch.float).t()
            p_pc[i, :, 3] = torch.from_numpy(self.scaler(E_modulus_dict.values())).type(torch.float).t()
            p_pc[i, :, 4] = torch.from_numpy(self.scaler(thermal_misfit_dict.values())).type(torch.float).t()
        return p_pc
    
    def fit(self):
        attention_data = torch.zeros(self.length, 2, 17, 6)
        attention_data[:, 0, :, :] = self.c_embedding()
        attention_data[:, 1, :, :] = self.p_embedding()
       
        particle_data = torch.zeros(self.length, 2, 6, 5)
        particle_data[:,0, :, :] = self.particle_embedding()
        particle_data[:,1, :, :] = self.p_particle_embedding()

        scaler = StandardScaler()
        scaler_values = scaler.fit_transform(self.process_value)

        scaler_params = {
        'mean': torch.from_numpy(scaler.mean_).float(),  # 均值
        'std': torch.from_numpy(scaler.scale_).float()}
        torch.save(scaler_params, r'D:\ML_AMC\Transfer_Workflow\Fine_tuning\Norm_Scalervalues.pt')
        scaler_values = torch.from_numpy(scaler_values).type(torch.float)
       
        return [attention_data, scaler_values, particle_data]
    


class Output_Normalizer:
    def __init__(self):
        self.means = []
        self.stds = []
    
    def fit_transform(self, data, sav_dir):
      if not isinstance(data, torch.Tensor):
          data = torch.from_numpy(data).float()
          
      normalized_data = data.clone()
      self.means = []
      self.stds = []
      
      for i in range(data.shape[1]):
          mean = data[:, i].mean()
          std = data[:, i].std()

          if std > 0:
              normalized_data[:, i] = (data[:, i] - mean) / std
          else:
              normalized_data[:, i] = 0
          
          self.means.append(mean.item())
          self.stds.append(std.item())

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
            denormalized_data[:, i] = data[:, i] * self.stds[i] + self.means[i]
        
        return denormalized_data

def main():
    df = pd.read_excel('.\composite_data.xlsx',sheet_name= 'Sheet1')
    length = len(df)

    elements = ['Si', 'Fe', 'Cu', 'B', 'Zn', 'Mn', 'Mg', 'Ti', 'V', 'Ni',
                'Ce', 'Cr', 'Sc', 'Sr', 'Zr', 'Li', 'Al']
    ceramic_particles = ['SiC', 'TiC', 'TiCN', 'Al3Ti', 'ZrB2', 'TiB2']

    at_value = torch.zeros(length, 17, 1)
    for i, element in enumerate(elements):
        if element in df.columns:
            at_value[:, i, 0] = torch.tensor(df[element].values, dtype=torch.float)

    particle_value = torch.zeros(length, 6, 1)
    for i, particle in enumerate(ceramic_particles):
        if particle in df.columns:
            particle_value[:, i, 0] = torch.tensor(df[particle].values, dtype=torch.float)

    process_columns = ['RT', 'PM', 'Particle_size', 'ST', 'AT', 'A_Time', 'HT', 'CTE', 'Orowan']
    process_value = df[process_columns].values

    preprocessor = Feature_Fusion(
        at_value=at_value,
        particle_value=particle_value,
        length=length,
        process_value=process_value
    )

    attention_data, scaler_values, particle_data = preprocessor.fit()

    torch.save(attention_data, './attention_data.pt')
    torch.save(scaler_values, './scaler_values.pt')
    torch.save(particle_data, './particle_data.pt')

if __name__ == "__main__":
    main()