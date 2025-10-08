"""
File Managers for mainting a controlled cache, selecting files for processing, outputting files etc.
"""




class CacheManager():
    """
    - ensures the cache exists and only contains the files it's supposed to
    - allows to preview the contents of the cache
    - keeps track of the size of the cache, gives warning when it gets too big
    - allows to empty the cache, warns of the consequences

    - maintains comprehensive file naming
    - saves new processing results the moment they are available (must be robust against sudden termination)
    - recovers results if processing is attempted for an already cached file
    - provides the files for displaying
    """
    pass


def ImageFileManager():
    """
    - selects single files for processing
        - ensures it's of a supported format, converts to the desired one
        - verifies metadata all adds up
    - selects full folders for processing
        - identifies all files that are fit for processing
        - gives warnings if the folder seems suspicious
        - locks and secures the folder for the duration of processing
    - saves image visualisations
        - provides some metadata for file origin
        - easy selection of target folder
        - autogenerates comprehensive naming, allows override
        - allows applying the same visualisation filter to multiple images
    """
    pass



