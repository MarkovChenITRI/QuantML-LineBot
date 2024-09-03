from sources import GET

class Market():
    def __init__(self):
        self.China = ['^HSI', '^HSCE', '000001.SS', '399001.SZ']
        self.Japan = ['^N225']
        self.UnitedStates = ['^DJI', '^GSPC', '^IXIC']
        self.Korea = ['^KS11']
        self.Taiwan = ['^TWII']
        self.Universe = ['BTC-USD', 'ETH-USD']
        self.debt = ['^TNX', 'GC=F']
        self.options = {"China": self.China, "Japan": self.Japan, "UnitedStates": self.UnitedStates, "Korea": self.Korea, "Taiwan": self.Taiwan, "Universe": self.Universe}

        self.Forex = ['GC=F', 'USDCNY=X', 'USDJPY=X', 'USDEUR=X', 'USDHKD=X', 'USDKRW=X']

        self.indexes = {"Forex": self.Forex, "Stock": self.China + self.Japan + self.UnitedStates + self.Korea + self.Taiwan + self.Universe, "Debt": self.debt}

    def get_data(self, pred_options = ["UnitedStates", "Taiwan", "Universe"], path='./data.csv', default="USDTWD=X"):
        print(f'[config.py] Market.get_data()')
        df = GET(default).copy()
        content = [f"Forex/{default}"]
        for cls in self.indexes:
            for code in self.indexes[cls]:
                if code != default:
                    for _ in range(3):
                        try:
                            temp_df = GET(code)
                            flag = False
                            for market in pred_options:
                                if code in self.options[market]:
                                    flag = True
                            if flag == True:
                                content.append(f'{cls}/{code} (Pred)')
                            else:
                                content.append(f'{cls}/{code}')
                            for col in temp_df:
                                if 'Pred' in col and flag==False:
                                    pass
                                else:
                                    df[col] = temp_df[col]
                            break
                        except:
                            pass
        print('[Summary from config.Market()]')
        for i in content:
            print(' -', i)
        options = {market: [] for market in pred_options}
        for i in df:
            if 'Pred' in i:
                code = i.split('/')[0]
                for market in options:
                    if code in self.options[market]:
                        options[market].append(i)
        return df.ffill().dropna(), options