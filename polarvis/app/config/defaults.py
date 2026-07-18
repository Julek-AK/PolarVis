"""
Default PolarVis settings
"""

DEFAULT_SETTINGS = {
    'settings_version': 1,
    'general': {},

    'display': {
        'theme': 'system',
        'autoscale': True,
    },

    'processing': {
        'use_gpu': True,
        'batch_size': 10,
    },

    'camera': {
        'channel_order': [90, 45, 135, 0],  # [deg]; top-left, top-right, bottom-left, bottom-right
        'size': [2048, 2448],  # (Height, Width)
    },

    'visualization': {
        'colormaps': {
            'intensity': 'grey',
            'dolp': 'viridis',
            'aop': 'hsv'
        },
        'legend_style': 'small',
    },

    'paths': {
        'open_file': 'default',
        'open_folder': 'default',
        'save_visualization': 'default',
        'export_data': 'default',
    },
}