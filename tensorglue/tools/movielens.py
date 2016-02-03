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
        zip_files = pd.Series(zfile.namelist())
        zip_file = zip_files[zip_files.str.endswith('ratings.dat')].iat[0]
        zdata = zfile.read(zip_file)
        delimiter = ';'
        zdata = zdata.replace('::', delimiter) # makes data compatible with pandas c-engine
        ml_data = pd.read_csv(StringIO(zdata), sep=delimiter, header=None, engine='c',
                                names=['userid', 'movieid', 'rating', 'timestamp'],
                                usecols=['userid', 'movieid', 'rating'])

        if get_genres:
            zip_file = zip_files[zip_files.str.endswith('movies.dat')].iat[0]
            with zfile.open(zip_file) as zdata:
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


def filter_short_head(data, threshold=0.01):
    short_head = data.groupby('movieid', sort=False)['userid'].nunique()
    short_head.sort_values(ascending=False, inplace=True)

    ratings_perc = short_head.cumsum()*1.0/short_head.sum()
    movies_perc = pd.np.arange(1, len(short_head)+1, dtype=pd.np.float64) / len(short_head)

    long_tail_movies = ratings_perc[movies_perc > threshold].index
    return long_tail_movies
