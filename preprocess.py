import pandas as pd

def preprocess_data(input_path, output_path):
    df = pd.read_csv(input_path)
    df = df[['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'BedroomAbvGr', 'FullBath', 'SalePrice']]
    df.fillna(df.mean(), inplace=True)
    df.to_csv(output_path, index=False)
    print("Data preprocessing completed.")

if __name__ == "__main__":
    preprocess_data("data/raw/housing.csv", "data/processed/cleaned_housing.csv")
