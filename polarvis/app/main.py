from config.settings import settings
from ..gui.main_window import run_main_window

 
def main():

    settings.load()
    
    run_main_window()
