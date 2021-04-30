
import pandas as pd

ratings = pd.read_csv(r'C:\Users\HANA\Downloads\ml-latest-small\ratings.csv')
ratings.to_csv('C:\\Users\\HANA\\Downloads\\ml-latest\\ml-latest\\ratings_noh.csv', index=False, header=False)
