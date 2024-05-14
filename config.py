
from sources import GET
import warnings
warnings.filterwarnings("ignore")

class Market():
    def __init__(self):
        self.China = ['^HSI', '^HSCE', '000001.SS', '399001.SZ']
        self.Japan = ['^N225']
        self.UnitedStates = ['^DJI', '^GSPC', '^IXIC']
        self.Korea = ['^KS11']
        self.Taiwan = ['^TWII']
        self.Universe = ['BTC-USD', 'ETH-USD']
        self.options = {"China": self.China, "Japan": self.Japan, "UnitedStates": self.UnitedStates, "Korea": self.Korea, "Taiwan": self.Taiwan, "Universe": self.Universe}

        self.Forex = ['GC=F', 'USDCNY=X', 'USDJPY=X', 'USDEUR=X', 'USDHKD=X', 'USDKRW=X']

        self.indexes = {"Forex": self.Forex, "Stock": self.China + self.Japan + self.UnitedStates + self.Korea + self.Taiwan + self.Universe}

    def get_data(self, path='./data.csv', default="USDTWD=X"):
        df = GET(default)
        content = [f"Forex/{default}"]
        for cls in self.indexes:
            for code in self.indexes[cls]:
                if code != default:
                    for _ in range(3):
                        try:
                            temp_df = GET(code)
                            content.append(f'{cls}/{code}')
                            for col in temp_df:
                                df[col] = temp_df[col]
                            break
                        except:
                            pass
        print('[Summary from config.Market()]\n')
        for i in content:
            print(' -', i)
        df.to_csv(path)
        return df