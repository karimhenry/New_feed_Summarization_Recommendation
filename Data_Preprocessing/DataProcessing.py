import os
import pandas as pd
# import random
# from itertools import chain


class DataPreprocess:
    def __init__(self):
        self.df_unpivoted_path = None
        self.__data = os.path.join(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))), 'Data')
        self.__processeddata_path = os.path.join(self.__data, 'processed')
        self.__rawdata_path = os.path.join(self.__data, 'raw')
        if not os.path.exists(self.__processeddata_path):
            os.mkdir(self.__processeddata_path)

    @staticmethod
    def _merge_dicts(x):
        """
        Merging auth multiple row into a single row

        Args:
        x:  a list of dictionaries to be merged for each auth

        Returns:
        dictionary: article as key and auth impression as value
        """
        return {k: v for d in x.dropna() for k, v in d.items()}

    def Preprocessing_Behaviors(self):
        """
        This Method is concerned with behavior.tsv applying some preprocessing and putting it
        in .csv format with following columns 'user_id', 'item_id'& 'rating'

        Args:
        DATA_DIR:  Directory for where raw data exists
        """
        # Loading Behaviour Dataset
        behaviors = pd.read_csv(self.__rawdata_path + '/behaviors.tsv', sep='\t', header=None,
                                names=['impression_id', 'user_id', 'time', 'history', 'impressions'])

        # Create An Impression list
        Imp_list = []

        # Split Each impression by space
        behaviors['impressions'] = behaviors['impressions'].str.split(' ')

        # For Every impression split by "-" and creat dictionary for each impression
        for i, each in enumerate(behaviors['impressions']):
            Imp_dict = {}
            for k in each:
                Imp = k.split('-')

                # If Impression exists we increment the value else we add it to new record
                if Imp[0] in Imp_dict:
                    Imp_dict[Imp[0]] += int(Imp[1])
                else:
                    Imp_dict[Imp[0]] = int(Imp[1])

            # Append each row to the impression list
            Imp_list.append(Imp_dict)

        # Update Behaviours dataframe impression column with the impression list of dictionaries
        behaviors['impressions'] = Imp_list

        # Number of Unique Articles
        # items = list(set(chain.from_iterable(sub.keys() for sub in Imp_list)))
        # print(f'We are dealing with {len(items)} unique article ')

        # Merging duplicated auth rows into single record
        beh_merged = behaviors[['user_id', 'history', 'impressions']]
        beh_merged = beh_merged.groupby(['user_id'], as_index=False).impressions.agg(self._merge_dicts)

        # Unpivot Behaviours Dataset
        df_unpivoted = pd.DataFrame(
            [[i, k, v] for i, d in beh_merged[['user_id', 'impressions']].values for k, v in d.items()],
            columns=['user_id', 'item_id', 'rating'])
        df_unpivoted.to_csv(os.path.join(self.__processeddata_path, 'df_unpivoted.csv'))

    def Preprocessing_News(self):
        """
        This Method is concerned with merging news.tsv with behaviors.tsv
        to get replace news article id in behaviors.tsv with subcategories
        applying some preprocessing and putting it
        in .csv format with following columns 'user_id', 'item_id'& 'rating'

        Args:
        DATA_DIR:  Directory for where raw data exists
        """
        self.df_unpivoted_path = os.path.join(self.__processeddata_path, 'df_unpivoted.csv')
        if not os.path.exists(self.df_unpivoted_path):
            self.Preprocessing_Behaviors()

        df_unpivoted = pd.read_csv(self.df_unpivoted_path)

        # Loading News Dataset
        news = pd.read_csv(self.__rawdata_path + '/news.tsv', sep='\t', header=None,
                           names=['News ID', 'Category', 'SubCategory', 'Title', 'Abstract', 'URL', 'Title Entities', 'Abstract Entities'])

        # Subset News Dataset
        news_df = news[['News ID', 'SubCategory']]

        # Merge the behaviours dataframe with news dataframe
        df_subcat = df_unpivoted.merge(news_df, how='left', left_on='item_id', right_on='News ID')

        # Check how many unique subcategories
        # SubCat_Num = df_subcat.SubCategory.nunique()
        # print(f'We are dealing with {SubCat_Num} unique Subcategories ')

        # Subset and rename the dataframe columns to match Matrix Factorization library ['user_id','item_id','rating']
        df_subcat = df_subcat[['user_id', 'SubCategory', 'rating']]
        df_subcat.rename(columns={'SubCategory': 'item_id'}, inplace=True)

        # Get Sum of clicked items by each auth
        df_totals = df_subcat.groupby(['user_id'], as_index=False)['rating'].sum()
        df_totals.rename(columns={'rating': 'total_clicked'}, inplace=True)

        # Get Sum of clicked items by each auth per subcategory
        df_subcat_totals = df_subcat.groupby(['user_id', 'item_id'], as_index=False)['rating'].sum()

        # Merge the Sum clicked by sub category with Total number clicked by auth
        SubCat_Df = df_subcat_totals.merge(df_totals, how='left', left_on='user_id', right_on='user_id')

        # Normalizing the ratings by dividing rating per total clicked articles
        # SubCat_Df['rating'].values[SubCat_Df['rating'] > 0] = 1
        SubCat_Df['rating'] = (SubCat_Df['rating'] / SubCat_Df['total_clicked'])

        # Subset the Subcat Dataframe
        SubCat_Df = SubCat_Df[['user_id', 'item_id', 'rating']]

        SubCat_Df.to_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Data', 'processed', 'SubCat_Df.csv'))

    def Preprocessing_Categories(self):
        """
        This Method is concerned with behavior.tsv applying some preprocessing and putting it
        in .csv format with following columns 'user_id', 'item_id'& 'rating'

        Args:
        DATA_DIR:  Directory for where raw data exists
        """
        self.df_unpivoted_path = os.path.join(self.__processeddata_path, 'df_unpivoted.csv')
        if not os.path.exists(self.df_unpivoted_path):
            self.Preprocessing_Behaviors()

        df_unpivoted = pd.read_csv(self.df_unpivoted_path)
        df_unpivoted = df_unpivoted[['item_id', 'rating']]
        df_unpivoted = df_unpivoted.groupby(['item_id']).agg('sum')
        # df_unpivoted = df_unpivoted[df_unpivoted['rating'] > 0]

        # Loading News Dataset
        news = pd.read_csv(self.__rawdata_path + '/news.tsv', sep='\t', header=None, names=['News ID', 'Category', 'SubCategory', 'Title', 'Abstract', 'URL', 'Title Entities', 'Abstract Entities'])

        # Subset News Dataset
        news_df = news[['News ID', 'Category', 'Title', 'Abstract', 'URL']]

        # Merge the behaviours dataframe with news dataframe
        df_subcat = df_unpivoted.merge(news_df, how='left', left_on='item_id', right_on='News ID')
        df_subcat = df_subcat.sort_values(by=['Category', 'rating'], ascending=[True, False])
        df_subcat = df_subcat.reindex(columns=['News ID',  'rating', 'Category', 'Title', 'Abstract', 'URL'])

        df_subcat.to_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Data', 'processed', 'Category_df.csv'), index=False)

    @staticmethod
    def Users():
        # Loading News Dataset
        path = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Data'), 'processed')
        df = pd.read_csv(path + '/df_unpivoted.csv').fillna("")
        users = df['user_id'].unique().tolist()  # 50000 Users
        return users

    @staticmethod
    def Categories():
        # Loading News Dataset
        path = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Data'), 'processed')
        df = pd.read_csv(path + '/Category_df.csv').fillna("")[['Category', 'rating']].groupby(['Category'])\
            .agg('sum').sort_values(by=['rating'], ascending=False).reset_index()
        categories = df['Category'].unique().tolist()  # 16 Users
        return categories


# ====Code Running====
# a = DataPreprocess()
# a.Preprocessing_Behaviors()
# a.Preprocessing_News()
# a.Preprocessing_Categories()
# print(random.choice(a.Users()))
# print(a.Categories())
