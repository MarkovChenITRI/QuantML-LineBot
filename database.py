import pandas as pd
from neo4j import GraphDatabase
from sources import Get_EPS, Get_PE, Get_Price, Get_Beta, Get_Sharpo
import yfinance as yf

def Indexes(USD):
    print(f'[database.py] Indexes()')
    AURA_CONNECTION_URI = "neo4j+s://6d2f5b5d.databases.neo4j.io"
    AURA_USERNAME = "neo4j"
    AURA_PASSWORD = "ZzZ6zeBZ1N7fB_UAHezzHY0LajAXj2z7tmI2HwHPWa8" #@param ["ZzZ6zeBZ1N7fB_UAHezzHY0LajAXj2z7tmI2HwHPWa8"]
    driver = GraphDatabase.driver(AURA_CONNECTION_URI, auth=(AURA_USERNAME, AURA_PASSWORD))
    analysis_table = []
    with driver.session() as session:
      utility = [i['u'] for i in session.run("MATCH (u:utility) RETURN u").data()]
      for u in utility:
        temp_df = yf.Ticker(u['code'])
        updated, price = Get_Price(temp_df, u['market'], USD)
        eps = Get_EPS(temp_df, u['market'], USD)
        pe_ratio = Get_PE(temp_df)
        beta = Get_Beta(temp_df)
        sharpo = Get_Sharpo(u['code'])
        analysis_table.append([u['name'], updated, price, eps, pe_ratio, beta, sharpo])
        
    res_df = pd.DataFrame(analysis_table, columns=['code', 'update', 'price', 'eps', 'pe_ratio', 'beta', 'sharpo'])
    res_df = res_df.replace(np.nan, None)
    for col in res_df.loc[:, ['price', 'eps', 'pe_ratio', 'beta', 'sharpo']]:
      res_df[col] = res_df[col].fillna(np.mean(res_df[col]))
    """
    with driver.session() as session:
      session.run("MATCH (u:utility WHERE u.name='{name}')  set u.update='{updated}' set u.price='{price}' set u.eps='{eps}'\
                  set u.pe_ratio='{pe_ratio}'  set u.beta='{beta}'  set u.sharpo='{sharpo}'".format(name=u['name'], 
                                                                                                    updated=updated,
                                                                                                    price=str(price), 
                                                                                                    eps=str(eps), 
                                                                                                    pe_ratio=str(pe_ratio),
                                                                                                    beta=str(beta),
                                                                                                    sharpo=str(sharpo),
                                                                                                    )
                    )"""
    return res_df