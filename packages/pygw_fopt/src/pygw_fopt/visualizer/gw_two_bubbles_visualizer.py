from ..utils.filter_dataframe import filter_dataframe
from .get_colors import get_colors_from_cmap
from bubble_gw.utils import sample
import matplotlib.ticker as ticker
import numpy as np

class GwTwoBubblesVisualizer():
    def __init__(self, df):
        self.df = df
        self.df_filtered = df
    
    def set_filter(self, filter_dict):
        self.df_filtered = filter_dataframe(self.df, filter_dict)
    
    def get_gw_averaged(self):
        w_peak = 2*np.pi / self.df_filtered.d.values[0]
        w_arr = sample(*self.df_filtered.ratio_w_sample.values[0]) * w_peak
        cos_thetak_arr = sample(*self.df_filtered.cos_thetak_sample.values[0])
        dcos = np.abs(cos_thetak_arr[1] - cos_thetak_arr[0])
        dE_dlogw_dcosthetak = self.df_filtered.dE_dlogw_dcosthetak.values[0]
        spectrum = 2 * np.trapezoid(dE_dlogw_dcosthetak, axis=0, dx=dcos)
        return (w_arr, spectrum)
    
    def plot_gw_averaged(self, fig, ax, **kwargs_plot):
        w_peak = 2*np.pi / self.df_filtered.d.values[0]
        w_arr = sample(*self.df_filtered.ratio_w_sample.values[0]) * w_peak
        cos_thetak_arr = sample(*self.df_filtered.cos_thetak_sample.values[0])
        dcos = np.abs(cos_thetak_arr[1] - cos_thetak_arr[0])
        dE_dlogw_dcosthetak = self.df_filtered.dE_dlogw_dcosthetak.values[0]
        spectrum = 2 * np.trapezoid(dE_dlogw_dcosthetak, axis=0, dx=dcos)
        ax.plot(w_arr, spectrum, **kwargs_plot)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\omega$")
        ax.set_ylabel(r"$\widehat{\Omega}_\text{GW} = \dfrac{dE_\text{GW}}{d\log \omega} = 2 \displaystyle\int_0^1 d \cos \theta_k \dfrac{dE_\text{GW}}{d \log \omega d \cos \theta_k}$")
        ax.yaxis.set_major_locator(ticker.LogLocator(numticks=999))
        ax.yaxis.set_minor_locator(ticker.LogLocator(numticks=999, subs="auto"))
        ax.grid(True, which="both", alpha=0.5)
        return fig, ax

    def plot_all_gw_angular(self, fig, ax):
        w_peak = 2*np.pi / self.df_filtered.d.values[0]
        w_arr = sample(*self.df_filtered.ratio_w_sample.values[0]) * w_peak
        cos_thetak_arr = sample(*self.df_filtered.cos_thetak_sample.values[0])
        colors = get_colors_from_cmap("inferno", len(cos_thetak_arr))
        for k, cos_thetak in enumerate(cos_thetak_arr):
            dE_dlogw_dcosthetak = self.df_filtered.dE_dlogw_dcosthetak.values[0][k, :]
            ax.plot(w_arr, dE_dlogw_dcosthetak, label=rf"$\cos \theta_k = {cos_thetak:.2f}$", color=colors[k])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\omega$")
        ax.set_ylabel(r"$\dfrac{dE_\text{GW}}{d \log \omega d \cos \theta_k}$")
        ax.yaxis.set_major_locator(ticker.LogLocator(numticks=999))
        ax.yaxis.set_minor_locator(ticker.LogLocator(numticks=999, subs="auto"))
        ax.grid(True, which="both", alpha=0.5)
        fig.legend(bbox_to_anchor=(0.95, 0.5), loc='center left', borderaxespad=0.)
        return fig, ax

    def plot_one_gw_angular(self, fig, ax, cos_thetak, **kwargs_plot):
        w_peak = 2*np.pi / self.df_filtered.d.values[0]
        w_arr = sample(*self.df_filtered.ratio_w_sample.values[0]) * w_peak
        cos_thetak_arr = sample(*self.df_filtered.cos_thetak_sample.values[0])
        closest_idx = np.argmin(np.abs(cos_thetak_arr - cos_thetak))
        dE_dlogw_dcosthetak = self.df_filtered.dE_dlogw_dcosthetak.values[0][closest_idx, :]
        ax.plot(w_arr, dE_dlogw_dcosthetak, label=rf"$\cos \theta_k = {cos_thetak_arr[closest_idx]:.2f}$, $(1+1)D$ simulation", **kwargs_plot)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\omega$")
        ax.set_ylabel(r"$\dfrac{dE_\text{GW}}{d \log \omega d \cos \theta_k}$")
        ax.yaxis.set_major_locator(ticker.LogLocator(numticks=999))
        ax.yaxis.set_minor_locator(ticker.LogLocator(numticks=999, subs="auto"))
        ax.grid(True, which="both", alpha=0.5)
        return fig, ax
