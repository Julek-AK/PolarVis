
 
def main():

    from ..app.config.settings import settings
    settings.load()

    from ..gui.main_window import run_main_window
    run_main_window()
