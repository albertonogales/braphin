from .importData import *
from .modelateData import *
from .tools import *


class Graph:
    """
    Pure EEG connectivity graph.

    Loads EEG data via MNE and computes connectivity matrices using the
    strategy pattern defined in eegraph.strategy.

    For multimodal support (EEG + fMRI) use ``braphin.Graph`` instead,
    which subclasses this class and adds the fMRI pipeline.
    """

    def __init__(self):
        self.data = None
        self.ch_names = None
        self.metadata = {}

    def load_data(self, path, exclude=[None], electrode_montage_path=None):
        """
        Load EEG data from *path*.

        Parameters
        ----------
        path : str
            Path to the EEG file (any format supported by MNE).
        exclude : list, optional
            Channel names to exclude.
        electrode_montage_path : str, optional
            Path to a montage file for channel renaming.
        """
        input_data = InputData(path, exclude)
        self.data = input_data.load()

        self.ch_names = list(self.data.ch_names)
        if electrode_montage_path is not None:
            self.ch_names = input_data.set_montage(electrode_montage_path)

        input_data.display_info(self.ch_names)

        self.metadata = {
            "sample_rate": self.data.info['sfreq'],
            "n_channels": self.data.info['nchan'],
            "duration": round(self.data.times.max(), 3),
        }

    def modelate(self, window_size, connectivity, bands=[None], threshold=None):
        """
        Compute a connectivity graph from the loaded EEG data.

        Parameters
        ----------
        window_size : int or None
            Window size in samples.  ``None`` uses the full recording.
        connectivity : str
            Name of the connectivity measure (see ``eegraph.tools.connectivity_measures``).
        bands : list, optional
            Frequency bands for band-limited measures.
        threshold : float, optional
            Edge weight threshold applied after connectivity estimation.

        Returns
        -------
        G : NetworkX Graph
        connectivity_matrix : np.ndarray
        """
        print('\033[1m' + 'Model Data.' + '\033[0m')
        print(search(connectivity_measures, connectivity))

        model_data = ModelData(
            self.data,
            self.ch_names,
            eval(search(connectivity_measures, connectivity)),
        )
        G, connectivity_matrix = model_data.connectivity_workflow(
            bands, window_size, threshold
        )
        return G, connectivity_matrix

    def visualize_html(self, graph, name, auto_open=True):
        fig = draw_graph(graph)
        fig.update_layout(title='', plot_bgcolor='white')
        fig.write_html(
            str(name) + '_plot.html',
            auto_open=auto_open,
            default_height='100%',
            default_width='100%',
        )

    def visualize_png(self, graph, name):
        fig = draw_graph(graph)
        fig.update_layout(title='', plot_bgcolor='white')
        fig.write_image(str(name) + '.png', format='png', height=1000, width=1800)
