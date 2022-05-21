import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import glob


def data_preprocessing_pipeline(df):
    df=df.drop('PassengerId',axis=1)
    df=df.drop('Name',axis=1)
    
    df['Cabin']=df['Cabin'].str.split('/')
    deck=[]
    number=[]
    side=[]
    for item in df['Cabin']:
        try:
            deck.append(item[0])
            number.append(int(item[1]))
            side.append(item[2])
        except:
            deck.append(float('nan'))
            number.append(float('nan'))
            side.append(float('nan'))
    
    df['cabin_deck']=pd.Series(deck)
    df['cabin_number']=pd.Series(number)
    df['cabin_side']=pd.Series(side)
    df[['cabin_deck','cabin_side']]=df[['cabin_deck','cabin_side']].fillna('Other')
    df['cabin_number']=df['cabin_number'].fillna(-1)

    
    
    df=df.drop('Cabin',axis=1)
    
    col_list=['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
    df[col_list]=df[col_list].fillna(0)
    
    df['VIP']=df['VIP'].fillna(False)
    

    n=df['Destination'].isna().sum()
    arr=np.random.random(size=(n,))
    lst=["" for i in range(n)]
    ser=pd.Series(lst)
    ser[arr<0.680433]='TRAPPIST-1e'
    ser[arr<0.680433+0.207063]='55 Cancri e'
    ser[arr>=0.680433+0.207063]='PSO J318.5-22'
    index = df['Destination'].index[df['Destination'].isna()]
    df['Destination'].iloc[index]=ser.values
    
    n=df['HomePlanet'].isna().sum()
    arr=np.random.random(size=(n,))
    lst=["" for i in range(n)]
    ser=pd.Series(lst)
    ser[arr<0.529391]='Earth'
    ser[arr<0.529391+0.245140]='Europa'
    ser[arr>=0.529391+0.245140]='Mars'
    index = df['HomePlanet'].index[df['HomePlanet'].isna()]
    df['HomePlanet'].iloc[index]=ser.values
    
    avg_age=df.groupby('HomePlanet').mean()['Age']
    planet_list=['Earth','Europa','Mars']
    
    for planet in planet_list:
        val=avg_age[planet]
        df['Age'][(df['Age'].isna() * df['HomePlanet']==planet)]=df['Age'][(df['Age'].isna() * df['HomePlanet']==planet)].fillna(val)
    
    df['CryoSleep']=df['CryoSleep'].fillna(False)
    
    df=pd.get_dummies(df,drop_first=True)
    
    df=df.astype('float')
    
    return df

def split_data(x,y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
    X_test,X_val,y_test,y_val=train_test_split(X_test,y_test,test_size=0.5)

    try:
        os.makedirs('./data/clean')
    except:
        files=glob.glob('./data/clean/*')
        for f in files:
            os.remove(f)

    np.savetxt('./data/clean/test_clean_feats.csv',X_test,delimiter=',')
    np.savetxt('./data/clean/test_clean_labels.csv',y_test,delimiter=',')

    np.savetxt('./data/clean/val_clean_feats.csv',X_val,delimiter=',')
    np.savetxt('./data/clean/val_clean_labels.csv',y_val,delimiter=',')

    np.savetxt('./data/clean/train_clean_feats.csv',X_train,delimiter=',')
    np.savetxt('./data/clean/train_clean_labels.csv',y_train,delimiter=',')


def clean_final_test_set(df):
    df=data_preprocessing_pipeline(df)
    X=df.values

    np.savetxt('./data/clean/final_test_set.csv',X,delimiter=',')

def preprocess():
    df=pd.read_csv('./data/train.csv')
    df=data_preprocessing_pipeline(df)
    x=df.drop('Transported',axis=1)
    y=df['Transported']
    split_data(x,y)

    df=pd.read_csv('./data/test.csv')
    clean_final_test_set(df)


if __name__=="__main__":
    preprocess()

    