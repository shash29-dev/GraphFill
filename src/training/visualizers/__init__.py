import logging

from src.training.visualizers.directory import DirectoryVisualizer,DirectoryVisualizerC2F

def make_visualizer(kind, **kwargs):
    logging.info(f'Make visualizer {kind}')

    if kind == 'cnn_gan':
        return DirectoryVisualizer(**kwargs)

    if kind == 'c2f':
        return DirectoryVisualizerC2F(**kwargs)

    raise ValueError(f'Unknown visualizer kind {kind}')
