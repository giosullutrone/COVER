import logging

# Set up logging
logging.basicConfig(filename='logs.txt', 
                    format='%(asctime)s %(levelname)s: %(message)s')

logger = logging.getLogger(__name__)