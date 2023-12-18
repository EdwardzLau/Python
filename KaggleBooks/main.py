import pandas as pd
import matplotlib.pyplot as plt

def main():
    pd.options.display.max_columns = 50
    df = pd.read_csv("books.csv") # reading csv

    # quick stats
    print(df.info())
    print(f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns")
    print(df.describe())

    # keeping only columns I will be using
    df = (df[ ['title', 'authors', 'average_rating', 'num_pages', 'ratings_count', 'text_reviews_count',
                 'language_code', 'publication_date', 'publisher'] ])

    # checking for duplicates
    print('The number of duplicated records is: ', df.duplicated().sum())

    # converting datatypes
    # print(df['num_pages'].value_counts())

    # Convert 'Column_with_numbers' to numeric values, coerce non-numeric to NaN
    numeric_column = pd.to_numeric(df['num_pages'], errors='coerce')

    # Extract non-numeric indexes (NaN) from the converted column
    non_numeric_indexes = df[pd.isnull(numeric_column)].index

    # Display the rows with non-numeric values
    print("\nNon-numeric indexes in 'num_pages':")
    print(non_numeric_indexes)

    # dropping these rows with errors
    df = df.drop(non_numeric_indexes)
    df = df.reset_index(drop= True)

    # converting data types and downcasting when needed
    df[['num_pages', 'ratings_count', 'text_reviews_count']] = df[['num_pages', 'ratings_count', 'text_reviews_count']].apply(pd.to_numeric, downcast = 'integer')
    df['average_rating'] = df['average_rating'].astype(float)

    # Convert the 'dates' column to datetime format with 'm/d/yy' format
    df['publication_date'] = pd.to_datetime(df['publication_date'], format='%m/%d/%y', errors='coerce')

    # find and remove rows with errors in date column
    errors = df['publication_date'] = pd.to_datetime(df['publication_date'], format='%m/%d/%y', errors='coerce')

    # Extract non-numeric indexes (NaN) from the converted column
    error_indexes = df[pd.isnull(errors)].index

    print("\nNaT indexes in 'publication_date':")
    print(error_indexes)
    df = df.drop(error_indexes)
    df = df.reset_index(drop= True)

    print("Earliest date:", df['publication_date'].min())
    print("Most recent date:", df['publication_date'].max())

    # Convert the 'dates' column to 'dd-mm-yyyy' format
    df['publication_date'] = df['publication_date'].dt.strftime('%m-%d-%Y')
    df['publication_date'] = pd.to_datetime(df['publication_date'])
    df = df[df['publication_date'] <= '2020-12-08'] # filter out dates after 2023-01-01
    df['publication_date'] = df['publication_date'].dt.date

    print("Cleaned most recent date:", df['publication_date'].max())

    print("Language Codes:", df.language_code.unique())

    # Convert the language_code column to numeric and filter out non-numeric values
    df = df[pd.to_numeric(df['language_code'], errors='coerce').isna()]

    # Remove leading/trailing whitespace from the 'language' column
    df['language_code'] = df['language_code'].str.strip()

    # Set values starting with 'en-' to 'eng'
    df.loc[df['language_code'].str.startswith('en-'), 'language_code'] = 'eng'

    # Take the first three characters of all other values
    df['language_code'] = df['language_code'].str[:3]

    print("Updated language Codes:", df.language_code.unique())

    # final checks
    print(df.info())
    print("Cleaned df shape:", df.shape)

    # Sort the dataframe by ratings_count in descending order and select the top 20
    # top_20_most_rated_books = df.sort_values('ratings_count', ascending=False).head(20)
    # plt.title("Top 20 rated books")
    # plt.xlabel("Book Title")
    # plt.xlabel("Number of Ratings")
    # plt.barh(top_20_most_rated_books['title'],top_20_most_rated_books['ratings_count'])
    # plt.show()
    #
    # language_counts = df['language_code'].value_counts()
    # plt.title("Distribution of Languages")
    # plt.xlabel("Count")
    # plt.ylabel("Language")
    # plt.barh(language_counts.index, language_counts)
    # plt.show()
    #
    # most_books = df.groupby('authors', as_index=False)['title'].count().sort_values('title', ascending=False).head(20)
    # plt.barh(most_books['authors'],most_books['title'])
    # plt.xlabel("Author")
    # plt.ylabel("Books Published")
    # plt.show()
    #
    # plt.hist(df['average_rating'])
    # plt.title('Distribution of Rating')
    # plt.xlabel("Rating")
    # plt.ylabel("Count")
    # plt.show()

    # converting dataframe to an Excel file
    df.to_excel('GoodReadsBooks.xlsx', sheet_name='Data')

if __name__ == '__main__':
     main()