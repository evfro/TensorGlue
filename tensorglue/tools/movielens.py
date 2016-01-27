import pandas as pd
from requests import get
from StringIO import StringIO
from pandas.io.common import ZipFile


def get_movielens_data(local_file=None, get_genres=False):
    '''Downloads movielens data and stores it in pandas dataframe.
    '''
    if not local_file:
        #print 'Downloading data...'
        zip_file_url = 'http://files.grouplens.org/datasets/movielens/ml-10m.zip'
        zip_response = get(zip_file_url)
        zip_contents = StringIO(zip_response.content)
        #print 'Done.'
    else:
        zip_contents = local_file
    
    #print 'Loading data into memory...'
    with ZipFile(zip_contents) as zfile:
        zdata = zfile.read('ml-10M100K/ratings.dat')
        delimiter = ';'
        zdata = zdata.replace('::', delimiter) # makes data compatible with pandas c-engine
        ml_data = pd.read_csv(StringIO(zdata), sep=delimiter, header=None, engine='c',
                                names=['userid', 'movieid', 'rating', 'timestamp'],
                                usecols=['userid', 'movieid', 'rating'])
        
        if get_genres:
            with zfile.open('ml-10M100K/movies.dat') as zdata:
                delimiter = '::'
                genres_data = pd.read_csv(zdata, sep=delimiter, header=None, engine='python',
                                            names=['movieid', 'movienm', 'genres'])
            
            ml_genres = split_genres(genres_data)
            ml_data = (ml_data, ml_genres)
    
    return ml_data

    
def split_genres(genres_data):    
    genres_split = genres_data['genres'].str.split('|')
    ml_genres = pd.merge(genres_data[['movieid', 'movienm']],
                         genres_split.apply(pd.Series),
                         left_index=True, right_index=True)
    ml_genres = ml_genres.set_index(['movieid', 'movienm']).stack().reset_index(level=2, drop=True)
    ml_genres = ml_genres.to_frame('genreid').reset_index()
    return ml_genres