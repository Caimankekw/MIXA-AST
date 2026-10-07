# dataloader_regression.py
import pandas as pd
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset


class NISQAFbankDataset(Dataset):
    def __init__(self, csv_file, audio_conf):
        self.df = pd.read_csv(csv_file).reset_index(drop=True)
        required = {'fbank_path', 'mos'}
        if not required.issubset(self.df.columns) or self.df.empty:
            raise ValueError('Feature CSV must be nonempty and contain fbank_path and mos.')
        self.audio_conf = audio_conf

        self.freqm = self.audio_conf.get('freqm', 0)
        self.timem = self.audio_conf.get('timem', 0)
        self.norm_mean = self.audio_conf.get('mean', 0.0)
        self.norm_std = self.audio_conf.get('std', 1.0)
        self.skip_norm = self.audio_conf.get('skip_norm', False)
        self.mode = self.audio_conf.get('mode', 'train')
        if not self.skip_norm and (not np.isfinite(self.norm_std) or self.norm_std <= 0):
            raise ValueError('Normalization std must be finite and positive.')

        print(f'--------------- NISQA regression dataloader ({self.mode}) ---------------')
        print(f'freqm={self.freqm}, timem={self.timem}')
        if self.skip_norm:
            print('skip normalization=True')
        else:
            print(f'use mean={self.norm_mean:.6f}, std={self.norm_std:.6f}')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        fbank = np.load(row['fbank_path']).astype(np.float32)   # [F, T]
        fbank = torch.from_numpy(fbank).float().transpose(0, 1) # -> [T, F]

        # SpecAug only for train
        if self.mode == 'train':
            fbank_aug = fbank.transpose(0, 1).unsqueeze(0)  # [1, F, T]
            if self.freqm > 0:
                fbank_aug = torchaudio.transforms.FrequencyMasking(self.freqm)(fbank_aug)
            if self.timem > 0:
                fbank_aug = torchaudio.transforms.TimeMasking(self.timem)(fbank_aug)
            fbank = fbank_aug.squeeze(0).transpose(0, 1)    # back to [T, F]

        if not self.skip_norm:
            fbank = (fbank - self.norm_mean) / (self.norm_std * 2.0)

        mos = torch.tensor(float(row['mos']), dtype=torch.float32)
        return fbank, mos