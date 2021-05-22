import pandas as pd


def load_data(data_filepath='../data/flavors_of_cacao.csv'):
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


def clean_bean_type(bean_type_col: pd.Series):
    """Fills in missing values for bean type and recodes less common types as other to reduce cardinality."""
    bean_type_col = bean_type_col.fillna("unknown").apply(lambda x: 'unknown' if x.strip() == '' else x)
    bean_type_col = bean_type_col.apply(collapse_bean_type)
    return bean_type_col


def collapse_bean_type(bean: str):
    if bean == 'unknown':
        label = 'unknown'
    elif 'blend' in bean.lower() or ('forasetero' in bean.lower() and 'criollo' in bean.lower()) or (
            'forasetero' in bean.lower() and 'trinitario' in bean.lower()):
        label = 'blend'
    elif 'forasetero' in bean.lower():
        label = 'forasetero'
    elif 'criollo' in bean.lower():
        label = 'criollo'
    elif 'trinitario' in bean.lower():
        label = 'trinitario'
    else:
        label = 'other'
    return label


def recode_bean_origin(bean_origin_col: pd.Series):
    bean_origin_col = bean_origin_col.fillna("unknown").apply(lambda x: 'unknown' if x.strip() == '' else x)
    bean_origin_col = bean_origin_col.apply(lambda x: 'multi' if ',' in x or '/' in x else x)
    bean_origin_col.apply(lambda x: x if bean_origin_col.value_counts()[x] > 20 else 'other')
    return bean_origin_col


if __name__ == '__main__':
    pd.set_option('display.max_rows', 100)
    data = load_data()
    data["bean_type"] = clean_bean_type(data['bean_type'])
    data.drop(columns=['REF'], inplace=True)
    data['cocoa_percent'] = data['cocoa_percent'].str.strip("%").astype(float) / 100
    data['bean_origin'] = recode_bean_origin(data['bean_origin'])
    data.to_pickle("../data/data_processed.pickle")
