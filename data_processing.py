import pandas as pd


def load_data(data_filepath='./flavors_of_cacao.csv'):
    data = pd.read_csv(data_filepath)
    # Simplify the column names
    # The reason for two versions is to handle Linux vs Windows newline special character conventions
    data = data.rename(columns={'Company\xa0\n(Maker-if known)': 'company',
                                'Specific Bean Origin\nor Bar Name': 'bar_origin',
                                'Review\nDate': 'review_year',
                                'Cocoa\nPercent': 'cocoa_percent',
                                'Company\nLocation': 'company_location',
                                'Bean\nType': 'bean_type',
                                'Broad Bean\nOrigin': 'bean_origin',

                                'Company\xa0\r\n(Maker-if known)': 'company',
                                'Specific Bean Origin\r\nor Bar Name': 'bar_origin',
                                'Review\r\nDate': 'review_year',
                                'Cocoa\r\nPercent': 'cocoa_percent',
                                'Company\r\nLocation': 'company_location',
                                'Bean\r\nType': 'bean_type',
                                'Broad Bean\r\nOrigin': 'bean_origin',
                                })
    data.index.name = 'id'
    return data


def toPercent(string):
    # This function will convert a percentage string into a decimal variable
    # The purpose is to convert cocoa_percent into a numerical value
    return (float(string.strip('%')) / 100)


def spaceToNan(datastring):
    # function to turn all cells with strings of one space character into a numpy null
    # if the input is not a string, it will return the original input
    # if the input is not a single space, it will return the input
    # This is meant to clean the bean_type feature
    if type(datastring) == str:
        if datastring.strip() == '':
            return np.nan
        else:
            return datastring.strip()
    else:
        return datastring


def nanToUnknown(datastring):
    # function to turn all cells with the pandas null value into a string reading 'unknown'
    # as stated below, the purpose is to relabel null values of bean_type as unknown
    # this will allow us to use that feature to see if it is significant
    # This has to be run after spaceToNan above
    if pd.isna(datastring):
        return 'unknown'

    else:
        return datastring

if __name__ == '__main__':
    pd.set_option('display.max_rows',100)
