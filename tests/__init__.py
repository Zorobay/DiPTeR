import logging
logging.basicConfig(level=logging.DEBUG)

mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)
