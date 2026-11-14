from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd


def render(output_dir: str | Path = 'research_outputs') -> list[Path]:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError('Plotting requires: python -m pip install matplotlib') from exc

    out=Path(output_dir)
    files=[]
    for stem,x,title,sets,ylabel in [
        ('vae_epoch_metrics','epoch','VAE reconstruction: observed-cell NMSE',
         ['train_nmse','val_nmse'],'NMSE vs training-scaler baseline'),
        ('persistence_epoch_metrics','epoch','CNN-LSTM: unweighted classification BCE',
         ['train_bce','val_bce'],'Binary cross entropy'),
        ('gan_step_metrics','step','WGAN-GP: training objectives (not test losses)',
         ['critic_loss','generator_loss','gradient_penalty'],'Adversarial objective')]:
        data=pd.read_csv(out/f'{stem}.csv')
        if data.empty: continue
        fig,ax=plt.subplots(figsize=(9,4.7))
        for col in sets:
            if col in data:
                ax.plot(data[x],data[col],label=col,linewidth=2)
        ax.set(title=title,xlabel=x.capitalize(),ylabel=ylabel)
        ax.grid(True,alpha=.25);ax.legend()
        fig.tight_layout()
        dest=out/(stem+'.png')
        fig.savefig(dest,dpi=170)
        plt.close(fig)
        files.append(dest)
    return files


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',default='research_outputs')
    args=parser.parse_args()
    for p in render(args.output_dir):
        print(p)

if __name__=='__main__':main()
