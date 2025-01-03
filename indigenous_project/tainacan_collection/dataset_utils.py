import unicodedata
import re

import pandas as pd

# Normalizing text and making it lower case, but preserving accentuation and optionally changing white spaces to underline
def text_normalization(text, is_title=False):
    text = text.lower()
    
    if is_title:
        text = unicodedata.normalize('NFKD', text)
        text = text.encode('ascii', 'ignore').decode('utf-8')
        text = re.sub(r'\s+', '_', text.strip())
        
    else:
        text = unicodedata.normalize('NFKC', text)
    
    return text


# Convert date to pandas data, even if it's in Portuguese
def date_conversion(date_str, port=True):
    months_translation = {
        'janeiro': 'January', 'Janeiro': 'January', 'jan': 'January', 'Jan': 'January',
        'fevereiro': 'February', 'Fevereiro': 'February', 'fev': 'February', 'Fev': 'February',
        'março': 'March', 'Março': 'March', 'mar': 'March', 'Mar': 'March',
        'abril': 'April', 'Abril': 'April', 'abr': 'April', 'Abr': 'April',
        'maio': 'May', 'Maio': 'May', 'mai': 'May', 'Mai': 'May',
        'junho': 'June', 'Junho': 'June', 'jun': 'June', 'Jun': 'June',
        'julho': 'July', 'Julho': 'July', 'jul': 'July', 'Jul': 'July',
        'agosto': 'August', 'Agosto': 'August', 'ago': 'August', 'Ago': 'August',
        'setembro': 'September', 'Setembro': 'September', 'set': 'September', 'Set': 'September',
        'outubro': 'October', 'Outubro': 'October', 'out': 'October', 'Out': 'October',
        'novembro': 'November', 'Novembro': 'November', 'nov': 'November', 'Nov': 'November',
        'dezembro': 'December', 'Dezembro': 'December', 'dez': 'December', 'Dez': 'December'
    }
    
    if port:
        month = date_str.split(' ')[0]
        date_str = date_str.replace(month, months_translation[month])

    return pd.to_datetime(date_str).date()