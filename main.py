from config import Market
import warnings
warnings.filterwarnings("ignore")

config = Market()
df = config.get_data()
df