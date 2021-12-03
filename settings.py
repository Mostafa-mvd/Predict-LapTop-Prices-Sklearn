import os

all_column_names = ['Company', 'Product', 'TypeName',
                    'Inches', 'ScreenResolution', 'Cpu', 'Ram', 'Memory',
                    'Gpu', 'OpSys', 'Weight', 'Price_euros']

feature_columns = ['Company', 'Product', 'TypeName',
                   'Inches', 'ScreenResolution', 'Cpu', 'Ram', 'Memory',
                   'Gpu', 'OpSys', 'Weight']

csv_file_path = f"{os.path.dirname(__file__)}/laptop_price.csv"
