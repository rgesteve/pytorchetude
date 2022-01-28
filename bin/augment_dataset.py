#!/usr/bin/env python

import sys
sys.path.append('../deep_tabular_augmentation')

import torch
from torch import nn

import pandas as pd
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, required=True)
parser.add_argument("--target_field", type=str, required=True)
parser.add_argument("--output_size", type=int, required=True, default=100)

import deep_tabular_augmentation as dta

def main():
    args = parser.parse_args()
    samples_to_generate = args.output_size
    print(f"Should be aiming to create {samples_to_generate} records.")

    input_file = args.input_file
    target_field = args.target_field
    if not os.path.exists(input_file):
        print(f"Cannot find specified path {input_file}.")
        sys.exit(1)
    df = pd.read_csv(input_file)
    columns = list(df.columns)
    columns.remove(target_field)

    mu,logvar = torch.load('embeddings.pth')
    sigma = torch.exp(logvar / 2)
    q = torch.distributions.Normal(mu.mean(axis=0), sigma.mean(axis=0))
    z = q.rsample(sample_shape=torch.Size([samples_to_generate]))
    D_in = len(columns)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = dta.Autoencoder(nn.Sequential(*dta.get_lin_layers(D_in, [50, 12, 12])),
                     nn.Sequential(*dta.get_lin_layers_rev(D_in, [50, 12, 12])),
                     latent_dim=5).to(device)

    state_from_file = torch.load('model.pth')
    model.load_state_dict(state_from_file)

    with torch.no_grad():
        pred = model.decode(z).cpu().numpy()

    output_df = pd.DataFrame(data=pred, columns=columns)
    output_df.to_csv('out.csv')

    print('Wrote output to out.csv')

if __name__ == '__main__':
    main()

