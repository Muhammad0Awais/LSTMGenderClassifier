import pandas as pd

def loadData(path):
  df = pd.read_csv(path, engine='python')

  df["target"] = df["Gender"].apply(lambda x : 1 if  x == "F" else 0)
  df.drop(["Gender"],axis=1,inplace=True)
  
  data, label = preprocessNames(df)

  return data, label
  
def preprocessNames(df):
  X = []
  y = []
  for name, gen in zip(df['Name'], df['target']):
    # if idx<3:
      y.append(gen)
      embed = []
      listName = list(name.lower())
      length = len(listName)
      for char in listName:
        embed.append(ord(char)-96) # My defined embedding
      if length < 15: # padding
        for i in range(length,15):
          embed.append(0)
      X.append(embed)
    # else:
  return X, y
